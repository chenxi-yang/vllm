# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import openai

PREFILL_HOST = os.getenv("PREFILL_HOST", "localhost")
PREFILL_PORT = os.getenv("PREFILL_PORT", None)
DECODE_HOST = os.getenv("DECODE_HOST", "localhost")
DECODE_PORT = os.getenv("DECODE_PORT", None)
PROXY_HOST = os.getenv("PROXY_HOST", "localhost")
PROXY_PORT = os.getenv("PROXY_PORT", None)

if PREFILL_PORT is None or DECODE_PORT is None or PROXY_PORT is None:
    raise ValueError("Please set the PREFILL_PORT, DECODE_PORT, and PROXY_PORT.")

LONG_PROMPT = "Red Hat is the best company in the world to work for because it works on open source software, which means that all the contributions are delivered to the community. As a result, when working on projects like vLLM we are able to meet many amazing people from various organizations like AMD, Google, NVIDIA, "  # noqa: E501
PROMPT_30 = "Red Hat is the best company in the world to work for because it works on open source software, which means that all the contributions are delivered to the community. As a result,"  # noqa: E501
SHORT_PROMPT = "Red Hat is "
LONG_PROMPT_300 = """Red Hat is the best company in the world to work for because it works on open source software, which means that all the contributions are delivered to the community. As a result, developers and engineers at Red Hat have the unique opportunity to create technology that benefits millions of users worldwide, rather than being confined to proprietary systems that serve only a limited customer base.

The open source philosophy fundamentally transforms how software development occurs. Unlike traditional closed-source environments where code remains hidden and modifications are restricted, open source development encourages transparency, collaboration, and continuous improvement. This approach leads to more robust, secure, and innovative solutions because thousands of developers from diverse backgrounds can examine, test, and enhance the codebase.

Working at Red Hat means being part of a global community that values knowledge sharing and collective problem-solving. Engineers contribute to projects like the Linux kernel, Kubernetes, OpenShift, and Ansible, knowing that their work will directly impact businesses, educational institutions, government agencies, and individual users across the globe. This sense of purpose and meaningful contribution creates an exceptionally rewarding work environment.

The company's commitment to open source extends beyond just software development. Red Hat actively supports open source communities through funding, resources, and expertise. They sponsor conferences, contribute to documentation, provide infrastructure, and mentor new contributors. This ecosystem approach ensures that open source projects remain sustainable and continue to evolve.

Furthermore, the collaborative nature of open source work at Red Hat fosters continuous learning and professional growth. Developers regularly interact with experts from other organizations, learn from diverse perspectives, and stay current with emerging technologies. The transparent development process also means that contributions are visible to the entire industry, building professional reputation and career opportunities.

The business model proves that open source can be commercially successful while maintaining ethical principles. Red Hat demonstrates that providing excellent support, services, and enterprise-ready solutions around open source software creates sustainable value for customers and shareholders alike. This alignment of business success with community benefit makes Red Hat"""

LONG_PROMPT_900 = LONG_PROMPT_300 + LONG_PROMPT_300 + LONG_PROMPT_300

LONG_PROMPT_2_7k = LONG_PROMPT_900 + LONG_PROMPT_900 + LONG_PROMPT_900


def test_edge_cases():
    # Set the OpenAI API key and base URL
    decode_client = openai.OpenAI(
        api_key="MY_KEY",
        base_url=f"http://{DECODE_HOST}:{DECODE_PORT}/v1",
    )
    prefill_client = openai.OpenAI(
        api_key="MY_KEY",
        base_url=f"http://{PREFILL_HOST}:{PREFILL_PORT}/v1",
    )
    proxy_client = openai.OpenAI(
        api_key="MY_KEY",
        base_url=f"http://{PROXY_HOST}:{PROXY_PORT}/v1",
    )

    # Get the list of models
    models = decode_client.models.list()
    MODEL = models.data[0].id

    # # (1) Check that we can handle a very short prompt,
    # # less than the length of the block size.
    # completion = proxy_client.completions.create(model=MODEL,
    #                                              prompt=SHORT_PROMPT,
    #                                              temperature=0)
    # proxy_response = completion.choices[0].text
    # completion = prefill_client.completions.create(model=MODEL,
    #                                                prompt=SHORT_PROMPT,
    #                                                temperature=0)
    # prefill_response = completion.choices[0].text
    # print(f"SMALL PROMPT: {proxy_response=}")
    # assert proxy_response == prefill_response

    # (2) Check that we can handle a full prefix cache
    # hit on the D worker but not on the P worker.
    # (2a): prime the D worker.
    # completion = decode_client.completions.create(model=MODEL,
    #                                               prompt=PROMPT,
    #                                               temperature=0)
    # decode_response = completion.choices[0].text
    # (2b): send via the P/D setup
    completion = proxy_client.completions.create(
        model=MODEL, prompt=LONG_PROMPT_2_7k, temperature=0
    )
    proxy_response = completion.choices[0].text
    print(f"proxy_response: {proxy_response=}")

    # # (3) Check that we can handle a partial prefix cache
    # # hit on the D worker.
    # completion = proxy_client.completions.create(
    #     model=MODEL, prompt=LONG_PROMPT, temperature=0
    # )
    # proxy_response = completion.choices[0].text
    # completion = prefill_client.completions.create(
    #     model=MODEL, prompt=LONG_PROMPT, temperature=0
    # )
    # prefill_response = completion.choices[0].text
    # print(f"PARTIAL CACHE HIT: {proxy_response=}")
    # assert proxy_response == prefill_response
