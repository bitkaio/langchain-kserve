/**
 * Utility functions for @langchain/kserve.
 *
 * Includes:
 * - Chat template formatters (ChatML, Llama-style)
 * - LangChain BaseMessage → OpenAI message conversion
 * - LangChain tool definition → OpenAI tool schema conversion
 */

import type {
  BaseMessage,
  MessageContent,
} from "@langchain/core/messages";
import { isAIMessage, isHumanMessage, isSystemMessage, isToolMessage } from "@langchain/core/messages";
import type {
  StructuredToolInterface,
} from "@langchain/core/tools";

import type { OpenAIChatMessage, OpenAIContentBlock, OpenAITool } from "./types.js";

// ============================================================
// Message content helpers
// ============================================================

/**
 * Extract a plain string from a message content value.
 * LangChain message content can be a string or an array of content blocks.
 * For KServe we flatten everything to a single string.
 */
function messageContentToString(content: MessageContent): string {
  if (typeof content === "string") {
    return content;
  }
  // Array of content blocks — concatenate text blocks
  return content
    .map((block) => {
      if (typeof block === "string") return block;
      if (block.type === "text") return block.text;
      return "";
    })
    .join("");
}

// ============================================================
// Multimodal content helpers
// ============================================================

/**
 * Convert a LangChain MessageContent value to an OpenAI-compatible content value.
 *
 * If the content is a plain string, it is returned as-is.
 * If the content is an array of blocks and any block is an image_url type,
 * the full array is returned as OpenAIContentBlock[].
 * If the content is a text-only array, a joined string is returned.
 *
 * @param content - LangChain message content
 * @returns OpenAI-compatible content value
 */
export function messageContentToOpenAI(
  content: MessageContent
): string | OpenAIContentBlock[] {
  if (typeof content === "string") {
    return content;
  }

  // Check if any block is an image_url
  const hasImage = content.some(
    (block) => typeof block !== "string" && block.type === "image_url"
  );

  if (hasImage) {
    return content.map((block): OpenAIContentBlock => {
      if (typeof block === "string") {
        return { type: "text", text: block };
      }
      if (block.type === "text") {
        return { type: "text", text: block.text };
      }
      if (block.type === "image_url") {
        const imgBlock = block as {
          type: "image_url";
          image_url: { url: string; detail?: "low" | "high" | "auto" };
        };
        return {
          type: "image_url",
          image_url: imgBlock.image_url,
        };
      }
      // Other block types — convert to text
      return { type: "text", text: String(block) };
    });
  }

  // Text-only array — join into a string
  return content
    .map((block) => {
      if (typeof block === "string") return block;
      if (block.type === "text") return block.text;
      return "";
    })
    .join("");
}

// ============================================================
// LangChain message → OpenAI message
// ============================================================

/**
 * Convert an array of LangChain BaseMessages to OpenAI-compatible message objects.
 *
 * @param messages - Array of LangChain messages
 * @returns Array of OpenAI-compatible message dicts
 */
export function convertMessagesToOpenAI(
  messages: BaseMessage[]
): OpenAIChatMessage[] {
  return messages.map((msg): OpenAIChatMessage => {
    if (isSystemMessage(msg)) {
      return {
        role: "system",
        content: messageContentToOpenAI(msg.content),
      };
    }

    if (isHumanMessage(msg)) {
      return {
        role: "user",
        content: messageContentToOpenAI(msg.content),
      };
    }

    if (isAIMessage(msg)) {
      const result: OpenAIChatMessage = {
        role: "assistant",
        content: messageContentToString(msg.content),
      };
      // Preserve tool calls if present
      if (
        "tool_calls" in msg &&
        Array.isArray(msg.tool_calls) &&
        msg.tool_calls.length > 0
      ) {
        result.tool_calls = msg.tool_calls.map((tc) => ({
          id: tc.id ?? "",
          type: "function" as const,
          function: {
            name: tc.name,
            arguments:
              typeof tc.args === "string"
                ? tc.args
                : JSON.stringify(tc.args),
          },
        }));
        // If there are tool calls, content should be null per OpenAI spec
        if (!result.content) {
          result.content = null;
        }
      }
      return result;
    }

    if (isToolMessage(msg)) {
      return {
        role: "tool",
        content: messageContentToString(msg.content),
        tool_call_id: msg.tool_call_id,
      };
    }

    // Generic / function messages — treat as assistant
    return {
      role: "assistant",
      content: messageContentToString(msg.content),
    };
  });
}

// ============================================================
// LangChain tool → OpenAI tool schema
// ============================================================

/**
 * Convert a LangChain StructuredToolInterface or plain object tool definition
 * to an OpenAI-compatible tool schema.
 *
 * @param tool - LangChain tool or raw OpenAI tool object
 * @returns OpenAI tool definition
 */
export function convertToolToOpenAI(
  tool: StructuredToolInterface | Record<string, unknown>
): OpenAITool {
  // Already an OpenAI-format tool
  if (
    "type" in tool &&
    tool.type === "function" &&
    "function" in tool
  ) {
    return tool as unknown as OpenAITool;
  }

  // LangChain StructuredToolInterface
  const structuredTool = tool as StructuredToolInterface;
  const schema =
    "schema" in structuredTool
      ? (structuredTool.schema as Record<string, unknown>)
      : {};

  return {
    type: "function",
    function: {
      name: structuredTool.name,
      description: structuredTool.description,
      parameters: schema,
    },
  };
}

// ============================================================
// Chat template formatters
// ============================================================

/** Represents a simple chat message for template formatting */
export interface SimpleChatMessage {
  role: string;
  content: string;
}

/**
 * Format messages using the ChatML template (default for Qwen models and most modern instruct models).
 *
 * Format:
 * ```
 * <|im_start|>system
 * You are a helpful assistant.<|im_end|>
 * <|im_start|>user
 * Hello!<|im_end|>
 * <|im_start|>assistant
 * ```
 *
 * @param messages - Array of chat messages
 * @returns Formatted prompt string
 */
export function formatChatML(messages: SimpleChatMessage[]): string {
  const parts: string[] = [];
  for (const msg of messages) {
    parts.push(`<|im_start|>${msg.role}\n${msg.content}<|im_end|>`);
  }
  parts.push("<|im_start|>assistant");
  return parts.join("\n");
}

/**
 * Format messages using the Llama 2 / Llama 3 [INST] template.
 *
 * Llama 2 format:
 * ```
 * <s>[INST] <<SYS>>
 * system message
 * <</SYS>>
 *
 * user message [/INST] assistant message </s><s>[INST] next user [/INST]
 * ```
 *
 * @param messages - Array of chat messages
 * @returns Formatted prompt string
 */
export function formatLlama(messages: SimpleChatMessage[]): string {
  const parts: string[] = [];
  let systemContent: string | undefined;
  let pendingUser: string | undefined;

  for (const msg of messages) {
    if (msg.role === "system") {
      systemContent = msg.content;
    } else if (msg.role === "user") {
      if (pendingUser !== undefined) {
        // Flush previous user turn without an assistant reply
        const userBlock = systemContent
          ? `<<SYS>>\n${systemContent}\n<</SYS>>\n\n${pendingUser}`
          : pendingUser;
        parts.push(`<s>[INST] ${userBlock} [/INST]`);
        systemContent = undefined;
      }
      pendingUser = msg.content;
    } else if (msg.role === "assistant") {
      if (pendingUser !== undefined) {
        const userBlock = systemContent
          ? `<<SYS>>\n${systemContent}\n<</SYS>>\n\n${pendingUser}`
          : pendingUser;
        parts.push(`<s>[INST] ${userBlock} [/INST] ${msg.content} </s>`);
        systemContent = undefined;
        pendingUser = undefined;
      }
    }
  }

  // Final user turn without assistant response
  if (pendingUser !== undefined) {
    const userBlock = systemContent
      ? `<<SYS>>\n${systemContent}\n<</SYS>>\n\n${pendingUser}`
      : pendingUser;
    parts.push(`<s>[INST] ${userBlock} [/INST]`);
  }

  return parts.join("\n");
}

/**
 * Format LangChain messages to a prompt string for V2 protocol inference.
 *
 * @param messages - LangChain messages
 * @param format   - Template format to use
 * @param customTemplate - Custom template string (ignored unless format === "custom")
 * @returns Formatted prompt string
 */
export function formatMessagesToPrompt(
  messages: BaseMessage[],
  format: "chatml" | "llama" | "custom" = "chatml",
  customTemplate?: string
): string {
  const simple: SimpleChatMessage[] = messages.map((msg) => ({
    role: isSystemMessage(msg)
      ? "system"
      : isHumanMessage(msg)
      ? "user"
      : isAIMessage(msg)
      ? "assistant"
      : "user",
    content: messageContentToString(msg.content),
  }));

  if (format === "llama") {
    return formatLlama(simple);
  }

  if (format === "custom") {
    if (!customTemplate) {
      throw new Error(
        'chatTemplate is set to "custom" but no customChatTemplate was provided'
      );
    }
    // Simple variable substitution — replace {{messages}} with JSON
    return customTemplate.replace(
      "{{messages}}",
      JSON.stringify(simple, null, 2)
    );
  }

  // Default: ChatML
  return formatChatML(simple);
}
