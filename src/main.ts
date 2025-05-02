import {config} from "dotenv";
import {ChatOpenAI} from "@langchain/openai";
import {BindToolsInput} from "@langchain/core/language_models/chat_models";
import {ChatBedrockConverse} from "@langchain/aws";
import {fromIni} from "@aws-sdk/credential-provider-ini";

const LLM_BEDROCK_MODEL = "us.meta.llama4-scout-17b-instruct-v1:0";
const OPEN_AI_MODEL = "llama-4-scout-17b-16e-instruct"

// This is for tracing purposes (see https://docs.arize.com/phoenix)
import "./instrumentation";


config();


main().catch((error) => {
    console.error("Error occurred:", error);
});

async function main() {
    console.log("Starting main...");

    let credentials = fromIni();
    const bedrockLLMClient = new ChatBedrockConverse({
        model: LLM_BEDROCK_MODEL,
        temperature: 0,
        credentials,
        region: "us-east-1",
        maxRetries: 3,
        verbose: process.env.DEBUG === "true",
        timeout: 60000
    });

    const cerebrasLLMClient = new ChatOpenAI({
        configuration: {
            baseURL: "https://api.cerebras.ai/v1"
        },
        apiKey: process.env.CREBRAS_API_KEY,
        model: OPEN_AI_MODEL,
        temperature: 0,
        maxRetries: 3,
        verbose: process.env.DEBUG === "true",
        timeout: 60000,
        metadata: {
            provider: "cerebras"
        }
    });

    const systemPrompt = `
You are an AI assistant designed to help users achieve specific goals. Your primary objective is to guide users towards accomplishing their tasks efficiently and effectively.
<agent_goal>
Your goal is to help user to write an email, specify the format and improve the structure.  
Ask for a topic, content. When you communicate with the user, be very precise and send the short messages, 2 sentences max.
</agent_goal>

- Conduct an internal analysis inside <internal_thought_process> tags. This analysis should not be visible in your final response to the user. In your analysis:
   - Identify and list key information from the user's query
   - Evaluate if the query is directly related to the specific agent goal
   - If related to the goal, brainstorm potential solutions or approaches (list at least 3)
   - Evaluate the pros and cons of each approach
   - Plan your response to guide the user effectively

- YOU MUST use SendMsgToUser to send a message to the user. This tool is the only way to communicate with the user.
`;
    const messages = [
        {
            content: `can you create and send an email to my friend derrick, wish him Happy Birthday`,
            role: "user"
        }
    ];

    const tools: BindToolsInput[] = [
        {
            type: "function",
            function: {
                strict: true,
                name: "SendMsgToUser",
                description: "Use when AI needs to send a message to the user",
                parameters: {
                    type: "object",
                    properties: {
                        msg_to_user: {
                            type: "string",
                            description: "The message to the user"
                        },
                        internal_thought_process: {
                            type: "string",
                            description: "Internal thought not visible to user"
                        }
                    }
                }
            }
        },
        {
            type: "function",
            function: {
                strict: true,
                name: "sendEmailTool",
                description: "Sends an email to the specified recipient(s)",
                parameters: {
                    type: "object",
                    properties: {
                        body: {
                            type: "string",
                            description: "Email content"
                        },
                        subject: {
                            type: "string",
                            description: "Email subject"
                        },
                        to: {
                            type: "array",
                            items: {
                                type: "object",
                                properties: {
                                    email: {
                                        type: "string",
                                        description: "recipient email "
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

    ]

    // Bedrock LLM
    const bedrockResponse = await bedrockLLMClient.invoke(
        [{role: "system", content: systemPrompt}, ...messages],
        {
            tools,
        }
    );
    console.log("bedrock response", JSON.stringify(bedrockResponse.tool_calls, null, 2));

    // Cerebras LLM
    const cerebrasResponse = await cerebrasLLMClient.invoke(
        [{role: "system", content: systemPrompt}, ...messages],
        {
            tools,
        }
    );
    console.log("cerebras response", JSON.stringify(cerebrasResponse.tool_calls, null, 2));
}

