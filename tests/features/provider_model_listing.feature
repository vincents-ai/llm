Feature: Provider Model Listing
  As a developer
  I want to list available models from different LLM providers
  So that I can discover and use models programmatically

  Background:
    Given I have initialized all providers

  Scenario: List Anthropic Claude models
    When I list models from the anthropic provider
    Then I should see the following models:
      | model_id                      | name                    | context_window | capabilities        |
      | claude-3-5-sonnet-20241022   | Claude 3.5 Sonnet       | 200000         | vision,tools,caching |
      | claude-3-opus-20240229       | Claude 3 Opus           | 200000         | vision,tools,caching |
      | claude-3-haiku-20240307      | Claude 3 Haiku          | 200000         | vision,tools         |
    And model pricing should include:
      | model_id                      | input_tokens_per_1k | completion_tokens_per_1k |
      | claude-3-5-sonnet-20241022   | 0.003               | 0.015                    |
      | claude-3-opus-20240229       | 0.015               | 0.075                    |
      | claude-3-haiku-20240307      | 0.00080             | 0.004                    |

  Scenario: List OpenAI models
    When I list models from the openai provider
    Then I should see the following models:
      | model_id           | name          | context_window | capabilities              |
      | gpt-4              | GPT-4         | 8192           | vision,tools,json         |
      | gpt-4-turbo        | GPT-4 Turbo   | 128000         | vision,tools,json         |
      | gpt-3.5-turbo      | GPT-3.5 Turbo | 4096           | tools,json                |
    And model pricing should include:
      | model_id           | input_tokens_per_1k | completion_tokens_per_1k |
      | gpt-4              | 0.03                | 0.06                     |
      | gpt-4-turbo        | 0.01                | 0.03                     |
      | gpt-3.5-turbo      | 0.0005              | 0.0015                   |

  Scenario: List Ollama local models
    Given Ollama server is running on http://localhost:11434
    When I list models from the ollama provider
    Then the model list should not be empty
    And each model should have:
      | field | requirement |
      | name  | not_empty   |
      | model_id | not_empty   |

  Scenario: Get detailed model information
    When I get model details for "claude-3-5-sonnet-20241022" from anthropic
    Then the model info should contain:
      | field               | type      |
      | id                  | string    |
      | name                | string    |
      | context_window      | number    |
      | capabilities        | array     |
      | pricing             | object    |
    And model capabilities should include:
      | capability  | supported |
      | vision      | true      |
      | tools       | true      |
      | json_mode   | true      |
      | caching     | true      |

  Scenario: Model capabilities consistency
    When I list models from "anthropic" provider
    And I list models from "openai" provider
    And I list models from "ollama" provider
    Then all models should have required fields:
      | field          |
      | id             |
      | name           |
      | context_window |
      | capabilities   |
      | pricing        |

  Scenario: Handle unknown provider gracefully
    When I try to list models from provider "unknown_provider"
    Then I should receive an error about unknown provider
    And the error message should contain "unknown_provider"

  Scenario: Filter models by capability
    When I list models from anthropic with filter "vision"
    Then I should only see models that support vision capability
    And the result should include "claude-3-5-sonnet-20241022"

  Scenario: Sort models by context window
    When I list models from openai and sort by context_window descending
    Then "gpt-4-turbo" should appear before "gpt-4"
    And "gpt-4" should appear before "gpt-3.5-turbo"
