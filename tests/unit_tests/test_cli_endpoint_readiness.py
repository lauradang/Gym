# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List
from unittest.mock import MagicMock

import requests
from omegaconf import DictConfig
from pytest import CaptureFixture, MonkeyPatch

import nemo_gym.cli.env
from nemo_gym.cli.env import (
    _collect_model_endpoint_urls,
    _endpoint_probe_url,
    _is_endpoint_listening,
    _wait_for_model_endpoints,
)


class _FakeClock:
    """Stand-in for `time` and `sleep` so the wait loop runs without real waiting."""

    def __init__(self) -> None:
        self._now = 0.0

    def __call__(self) -> float:
        return self._now

    def sleep(self, seconds: float) -> None:
        self._now += seconds


def _config(**model_servers) -> DictConfig:
    return DictConfig(
        {
            "dry_run": False,
            "head_server": {"host": "127.0.0.1", "port": 11000},
            **model_servers,
        }
    )


class TestCollectModelEndpointUrls:
    def test_finds_every_base_url_key_spelling(self) -> None:
        # openai_model uses openai_base_url, inference_provider and vllm_model use base_url, the
        # Anthropic path uses anthropic_base_url.
        global_config_dict = _config(
            policy_model={
                "responses_api_models": {
                    "openai_model": {"entrypoint": "app.py", "openai_base_url": "https://api.openai.com/v1"}
                }
            },
            vllm={
                "responses_api_models": {
                    "vllm_model": {"entrypoint": "app.py", "base_url": "http://localhost:8000/v1"}
                }
            },
            claude={
                "responses_api_models": {
                    "anthropic_model": {"entrypoint": "app.py", "anthropic_base_url": "http://localhost:9000"}
                }
            },
        )

        assert [
            "https://api.openai.com/v1",
            "http://localhost:8000/v1",
            "http://localhost:9000",
        ] == _collect_model_endpoint_urls(global_config_dict)

    def test_finds_hardcoded_provider_urls(self) -> None:
        """Most inference_provider configs carry the URL literally rather than interpolating."""
        global_config_dict = _config(
            fireworks={
                "responses_api_models": {
                    "inference_provider": {
                        "entrypoint": "app.py",
                        "base_url": "https://api.fireworks.ai/inference/v1",
                    }
                }
            },
        )

        assert ["https://api.fireworks.ai/inference/v1"] == _collect_model_endpoint_urls(global_config_dict)

    def test_deduplicates_and_skips_non_model_servers(self) -> None:
        global_config_dict = _config(
            policy_model={
                "responses_api_models": {
                    "openai_model": {"entrypoint": "app.py", "openai_base_url": "http://localhost:8000/v1"}
                }
            },
            judge_model={
                "responses_api_models": {
                    "openai_model": {"entrypoint": "app.py", "openai_base_url": "http://localhost:8000/v1"}
                }
            },
            # Gym's own servers are already covered by wait_for_spinup and must not be probed here.
            mcqa={"resources_servers": {"mcqa": {"entrypoint": "app.py", "host": "127.0.0.1", "port": 13604}}},
            agent={"responses_api_agents": {"simple_agent": {"entrypoint": "app.py", "agent_base_url": "http://x:1"}}},
        )

        assert ["http://localhost:8000/v1"] == _collect_model_endpoint_urls(global_config_dict)

    def test_no_model_servers_configured(self) -> None:
        assert [] == _collect_model_endpoint_urls(_config())

    def test_ignores_unset_and_non_string_values(self) -> None:
        global_config_dict = _config(
            policy_model={"responses_api_models": {"openai_model": {"entrypoint": "app.py", "openai_base_url": ""}}},
        )

        assert [] == _collect_model_endpoint_urls(global_config_dict)


class TestEndpointProbe:
    def test_probe_url(self) -> None:
        assert "http://localhost:8000/v1/models" == _endpoint_probe_url("http://localhost:8000/v1")
        assert "http://localhost:8000/v1/models" == _endpoint_probe_url("http://localhost:8000/v1/")
        # Not an OpenAI-shaped URL, so probe the root.
        assert "http://localhost:9000" == _endpoint_probe_url("http://localhost:9000")

    def test_any_http_answer_counts_as_listening(self, monkeypatch: MonkeyPatch) -> None:
        """Listening is the bar, not healthy. Requiring a 200 would reject endpoints needing auth."""
        for status_code in (200, 401, 404, 500):
            get_mock = MagicMock(return_value=MagicMock(status_code=status_code))
            monkeypatch.setattr(nemo_gym.cli.env.requests, "get", get_mock)
            assert _is_endpoint_listening("http://localhost:8000/v1")

    def test_connection_errors_count_as_not_listening(self, monkeypatch: MonkeyPatch) -> None:
        for exc in (requests.exceptions.ConnectionError(), requests.exceptions.Timeout()):
            monkeypatch.setattr(nemo_gym.cli.env.requests, "get", MagicMock(side_effect=exc))
            assert not _is_endpoint_listening("http://localhost:8000/v1")

    def test_unprobeable_urls_do_not_block_startup(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(
            nemo_gym.cli.env.requests, "get", MagicMock(side_effect=requests.exceptions.MissingSchema())
        )
        assert _is_endpoint_listening("not-a-url")


class TestWaitForModelEndpoints:
    def _listening_after(self, num_failures: int, calls: List[str]) -> MagicMock:
        def probe(base_url: str, timeout_seconds: float = 5.0) -> bool:
            calls.append(base_url)
            return len(calls) > num_failures

        return MagicMock(side_effect=probe)

    def test_endpoint_already_up_passes_quietly(self, monkeypatch: MonkeyPatch, capsys: CaptureFixture) -> None:
        monkeypatch.setattr(nemo_gym.cli.env, "_is_endpoint_listening", MagicMock(return_value=True))

        assert [] == _wait_for_model_endpoints(["http://localhost:8000/v1"], timeout_seconds=600)
        assert "" == capsys.readouterr().out

    def test_endpoint_comes_up_on_the_third_poll(self, monkeypatch: MonkeyPatch) -> None:
        calls: List[str] = []
        monkeypatch.setattr(nemo_gym.cli.env, "_is_endpoint_listening", self._listening_after(3, calls))
        clock = _FakeClock()

        assert [] == _wait_for_model_endpoints(
            ["http://localhost:8000/v1"], timeout_seconds=600, monotonic=clock, sleep_fn=clock.sleep
        )
        assert 4 == len(calls)

    def test_endpoint_never_comes_up(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(nemo_gym.cli.env, "_is_endpoint_listening", MagicMock(return_value=False))
        clock = _FakeClock()

        unreachable = _wait_for_model_endpoints(
            ["http://localhost:8000/v1"], timeout_seconds=30, monotonic=clock, sleep_fn=clock.sleep
        )

        assert ["http://localhost:8000/v1"] == unreachable
        assert clock() >= 30

    def test_zero_timeout_skips_the_check(self, monkeypatch: MonkeyPatch) -> None:
        probe_mock = MagicMock(return_value=False)
        monkeypatch.setattr(nemo_gym.cli.env, "_is_endpoint_listening", probe_mock)

        assert [] == _wait_for_model_endpoints(["http://localhost:8000/v1"], timeout_seconds=0)
        probe_mock.assert_not_called()

    def test_no_endpoints_is_a_no_op(self, monkeypatch: MonkeyPatch) -> None:
        probe_mock = MagicMock(return_value=False)
        monkeypatch.setattr(nemo_gym.cli.env, "_is_endpoint_listening", probe_mock)

        assert [] == _wait_for_model_endpoints([], timeout_seconds=600)
        probe_mock.assert_not_called()

    def test_only_the_endpoints_still_down_are_reported(self, monkeypatch: MonkeyPatch) -> None:
        up = "http://up:8000/v1"
        down = "http://down:8000/v1"
        monkeypatch.setattr(
            nemo_gym.cli.env, "_is_endpoint_listening", MagicMock(side_effect=lambda url, **_: url == up)
        )
        clock = _FakeClock()

        assert [down] == _wait_for_model_endpoints(
            [up, down], timeout_seconds=30, monotonic=clock, sleep_fn=clock.sleep
        )
