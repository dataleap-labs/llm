package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/sashabaranov/go-openai"
)

// OpenAILLM implements the LLM interface for OpenAI
type OpenAILLM struct {
	client *openai.Client
}

type OpenAIModel string

// NewOpenAILLM creates a new OpenAI LLM client
func NewOpenAILLM(apiKey string) *OpenAILLM {
	client := openai.NewClient(apiKey)
	return &OpenAILLM{client: client}
}

// NewOpenAILLMWithBaseURL creates a new OpenAI LLM client with a custom base URL
func NewOpenAILLMWithBaseURL(apiKey string, baseURL string) *OpenAILLM {
	config := openai.DefaultConfig(apiKey)
	config.BaseURL = baseURL
	client := openai.NewClientWithConfig(config)
	return &OpenAILLM{client: client}
}

func NewAzureLLM(apiKey string, azureOpenAIEndpoint string) *OpenAILLM {
	// The latest API versions, including previews, can be found here:
	// https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#rest-api-versioning
	config := openai.DefaultAzureConfig(apiKey, azureOpenAIEndpoint)
	config.APIVersion = "2023-05-15" // optional update to latest API version

	//If you use a deployment name different from the model name, you can customize the AzureModelMapperFunc function
	//config.AzureModelMapperFunc = func(model string) string {
	//    azureModelMapping := map[string]string{
	//        "gpt-3.5-turbo":"your gpt-3.5-turbo deployment name",
	//    }
	//    return azureModelMapping[model]
	//}

	client := openai.NewClientWithConfig(config)
	return &OpenAILLM{client: client}
}

// convertToOpenAIMessages converts our generic Message type to OpenAI's message type
func convertToOpenAIMessages(messages []InputMessage) []openai.ChatCompletionMessage {
	openAIMessages := make([]openai.ChatCompletionMessage, 0, len(messages))

	for _, msg := range messages {

		var role string
		switch msg.Role {
		case RoleUser:
			role = openai.ChatMessageRoleUser
		case RoleAssistant:
			role = openai.ChatMessageRoleAssistant
		case RoleTool:
			role = openai.ChatMessageRoleTool
		}

		var openAIMsg openai.ChatCompletionMessage
		openAIMsg.Role = role
		if msg.Role == RoleTool {
			openAIMsg.Content = msg.ToolResults[0].Result
			openAIMsg.ToolCallID = msg.ToolResults[0].ToolCallID
		} else {
			openAIMsg.MultiContent = convertOpenAIMessageContent(msg.MultiContent)
			openAIMsg.ToolCalls = convertToOpenAIToolsCalls(msg.ToolCalls)
		}

		openAIMessages = append(openAIMessages, openAIMsg)
	}
	return openAIMessages
}

func convertOpenAIMessageContent(content []ContentPart) []openai.ChatMessagePart {
	multiContent := make([]openai.ChatMessagePart, 0, len(content))
	for _, part := range content {
		switch part.Type {
		case ContentTypeText:
			multiContent = append(multiContent, openai.ChatMessagePart{
				Type: openai.ChatMessagePartTypeText,
				Text: part.Text,
			})
		case ContentTypeImage:
			imageURL := "data:" + part.MediaType + ";base64," + part.Data
			multiContent = append(multiContent, openai.ChatMessagePart{
				Type: openai.ChatMessagePartTypeImageURL,
				ImageURL: &openai.ChatMessageImageURL{
					URL:    imageURL,
					Detail: "high",
				},
			})
		}
	}
	return multiContent
}

// convertFromOpenAIMessage converts OpenAI's message type to our generic Message type
func convertFromOpenAIMessage(msg openai.ChatCompletionMessage) OutputMessage {

	var content string
	if len(msg.MultiContent) > 0 {
		// Handle multi-content messages
		var textParts []string
		for _, part := range msg.MultiContent {
			if part.Type == openai.ChatMessagePartTypeText {
				textParts = append(textParts, part.Text)
			}
		}
		content = strings.Join(textParts, "")
	} else {
		// Handle regular content
		content = msg.Content
	}

	return OutputMessage{
		Role:      Role(msg.Role),
		Content:   content,
		ToolCalls: convertFromOpenAIToolCalls(msg.ToolCalls),
	}
}

// convertToOpenAITools converts our generic Tool type to OpenAI's tool type
func convertToOpenAITools(tools []Tool) []openai.Tool {
	if len(tools) == 0 {
		return nil
	}

	openAITools := make([]openai.Tool, len(tools))
	for i, tool := range tools {
		def := openai.FunctionDefinition{
			Name:        tool.Function.Name,
			Description: tool.Function.Description,
			Parameters:  tool.Function.Parameters,
		}
		openAITools[i] = openai.Tool{
			Type:     openai.ToolTypeFunction,
			Function: &def,
		}
	}
	return openAITools
}

func convertToOpenAIToolsCalls(tools []ToolCall) []openai.ToolCall {
	if len(tools) == 0 {
		return nil
	}

	openAITools := make([]openai.ToolCall, len(tools))
	for i, tool := range tools {
		def := openai.FunctionCall{
			Name:      tool.Function.Name,
			Arguments: tool.Function.Arguments,
		}
		openAITools[i] = openai.ToolCall{
			ID:       tool.ID,
			Type:     openai.ToolTypeFunction,
			Function: def,
		}
	}

	return openAITools
}

// convertFromOpenAIToolCalls converts OpenAI's tool calls to our generic type
func convertFromOpenAIToolCalls(toolCalls []openai.ToolCall) []ToolCall {
	if len(toolCalls) == 0 {
		return nil
	}

	calls := make([]ToolCall, len(toolCalls))
	for i, call := range toolCalls {

		toolfunc := ToolCallFunction{
			Name:      call.Function.Name,
			Arguments: call.Function.Arguments,
		}

		calls[i] = ToolCall{
			ID:       call.ID,
			Type:     string(call.Type),
			Function: toolfunc,
		}
	}
	return calls
}

// CreateChatCompletion implements the LLM interface for OpenAI
func (o *OpenAILLM) CreateChatCompletion(ctx context.Context, req ChatCompletionRequest) (ChatCompletionResponse, error) {

	// check if model is compatible with OpenAI
	if !o.isSupported(req.Model) {
		return ChatCompletionResponse{}, fmt.Errorf("model %s is not available", req.Model)
	}

	topP := float32(1)
	if req.TopP != nil {
		topP = *req.TopP
	}

	// Set system prompt if provided
	var messages []openai.ChatCompletionMessage
	if req.SystemPrompt != nil {
		messages = append(messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleSystem,
			Content: *req.SystemPrompt,
		})
	}

	inputMessages := convertToOpenAIMessages(req.Messages)
	messages = append(messages, inputMessages...)

	openAIReq := openai.ChatCompletionRequest{
		Model:               string(req.Model), // TODO: convert model name to OpenAI model name
		Messages:            messages,
		Temperature:         req.Temperature,
		N:                   1,
		TopP:                topP,
		Stop:                []string{},
		Tools:               convertToOpenAITools(req.Tools),
		Stream:              false,
		MaxCompletionTokens: req.MaxTokens,
	}

	if req.JSONMode {
		openAIReq.ResponseFormat = &openai.ChatCompletionResponseFormat{
			Type: openai.ChatCompletionResponseFormatTypeJSONObject,
		}
	}

	if req.Model == ModelO3Mini {
		openAIReq.ReasoningEffort = "high"
	}

	resp, err := o.client.CreateChatCompletion(ctx, openAIReq)
	if err != nil {
		return ChatCompletionResponse{}, err
	}

	choices := make([]Choice, len(resp.Choices))
	for i, c := range resp.Choices {
		msg := convertFromOpenAIMessage(c.Message)
		msg.ToolCalls = convertFromOpenAIToolCalls(c.Message.ToolCalls)
		finishReason, err := convertFromOpenAIFinishReason(c.FinishReason)
		if err != nil {
			return ChatCompletionResponse{}, err
		}
		choices[i] = Choice{
			Index:        c.Index,
			Message:      msg,
			FinishReason: finishReason,
		}
	}

	return ChatCompletionResponse{
		ID:      resp.ID,
		Choices: choices,
		Usage: Usage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		},
	}, nil
}

func (o *OpenAILLM) isSupported(model Model) bool {

	switch model {
	case ModelO3Mini:
		return true
	case ModelGPT4o:
		return true
	case ModelGPT4oMini:
		return true
	default:
		return false
	}
}

// openAIStreamWrapper wraps the OpenAI stream
type openAIStreamWrapper struct {
	stream          *openai.ChatCompletionStream
	currentToolCall *ToolCall
	toolCallBuffer  map[string]*ToolCall
}

func newOpenAIStreamWrapper(stream *openai.ChatCompletionStream) *openAIStreamWrapper {
	return &openAIStreamWrapper{
		stream:         stream,
		toolCallBuffer: make(map[string]*ToolCall),
	}
}

func (w *openAIStreamWrapper) Recv() (ChatCompletionResponse, error) {
	resp, err := w.stream.Recv()
	if err != nil {
		if err == io.EOF {
			return ChatCompletionResponse{}, err
		}
		var openAIErr *openai.APIError
		if errors.As(err, &openAIErr) {
			return ChatCompletionResponse{}, fmt.Errorf("OpenAI API error: %s - %s", openAIErr.Code, openAIErr.Message)
		}
		return ChatCompletionResponse{}, fmt.Errorf("stream receive failed: %w", err)
	}

	choices := make([]Choice, len(resp.Choices))
	for i, c := range resp.Choices {

		// Handle tool calls in delta
		var toolCalls []ToolCall
		if len(c.Delta.ToolCalls) > 0 {
			toolCalls = make([]ToolCall, 0)
			for _, tc := range c.Delta.ToolCalls {
				// Get or create tool call buffer
				toolCall, exists := w.toolCallBuffer[tc.ID]
				if !exists {
					if tc.ID == "" {
						// Skip empty IDs but accumulate arguments if present
						if tc.Function.Arguments != "" && w.currentToolCall != nil {
							w.currentToolCall.Function.Arguments += tc.Function.Arguments

							// Try to parse the arguments to verify if it's complete JSON
							if isValidJSON(w.currentToolCall.Function.Arguments) {
								toolCalls = append(toolCalls, *w.currentToolCall)
								delete(w.toolCallBuffer, w.currentToolCall.ID)
								w.currentToolCall = nil
							}
						}
						continue
					}

					// Create new tool call
					toolCall = &ToolCall{
						ID:   tc.ID,
						Type: string(tc.Type),
						Function: ToolCallFunction{
							Name:      tc.Function.Name,
							Arguments: "",
						},
					}
					w.toolCallBuffer[tc.ID] = toolCall
					w.currentToolCall = toolCall
				}

				// Accumulate tool call data
				if tc.Function.Name != "" {
					toolCall.Function.Name = tc.Function.Name
				}
				if tc.Function.Arguments != "" {
					toolCall.Function.Arguments += tc.Function.Arguments

					// Check if we have complete JSON
					if isValidJSON(toolCall.Function.Arguments) {
						toolCalls = append(toolCalls, *toolCall)
						delete(w.toolCallBuffer, tc.ID)
						if w.currentToolCall != nil && w.currentToolCall.ID == tc.ID {
							w.currentToolCall = nil
						}
					}
				}
			}
		}

		// Create the message with accumulated content
		message := OutputMessage{
			Role:      Role(c.Delta.Role),
			Content:   c.Delta.Content,
			ToolCalls: toolCalls,
		}

		finishReason, err := convertFromOpenAIFinishReason(c.FinishReason)
		if err != nil {
			return ChatCompletionResponse{}, err
		}

		choices[i] = Choice{
			Index:        c.Index,
			Message:      message,
			FinishReason: finishReason,
		}
	}

	return ChatCompletionResponse{
		ID:      resp.ID,
		Choices: choices,
	}, nil
}

// Helper function to check if a string is valid JSON
func isValidJSON(s string) bool {
	var js map[string]interface{}
	return json.Unmarshal([]byte(s), &js) == nil
}

func convertFromOpenAIFinishReason(reason openai.FinishReason) (FinishReason, error) {

	switch reason {
	case openai.FinishReasonToolCalls:
		return FinishReasonToolCalls, nil
	case openai.FinishReasonStop:
		return FinishReasonStop, nil
	case openai.FinishReasonLength:
		return FinishReasonMaxTokens, nil
	case openai.FinishReasonFunctionCall:
		return FinishReasonToolCalls, nil
	case openai.FinishReasonContentFilter:
		return FinishReasonStop, nil
	case openai.FinishReasonNull:
		return FinishReasonNull, nil
	case "":
		return FinishReasonNull, nil
	default:
		return FinishReason(""), fmt.Errorf("default case reason: %s", reason)
	}
}

func (w *openAIStreamWrapper) Close() error {
	return w.stream.Close()
}

// CreateChatCompletionStream implements the LLM interface for OpenAI streaming
func (o *OpenAILLM) CreateChatCompletionStream(ctx context.Context, req ChatCompletionRequest) (ChatCompletionStream, error) {

	// check if model is compatible with OpenAI
	if !o.isSupported(req.Model) {
		return nil, fmt.Errorf("model %s is not available", req.Model)
	}
	topP := float32(1)
	if req.TopP != nil {
		topP = *req.TopP
	}

	// Set system prompt if provided
	var messages []openai.ChatCompletionMessage
	if req.SystemPrompt != nil {
		messages = append(messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleSystem,
			Content: *req.SystemPrompt,
		})
	}

	inputMessages := convertToOpenAIMessages(req.Messages)
	messages = append(messages, inputMessages...)

	openAIReq := openai.ChatCompletionRequest{
		Model:               string(req.Model), // TODO: convert model name
		Messages:            messages,
		Temperature:         req.Temperature,
		N:                   1,
		Stop:                []string{},
		Tools:               convertToOpenAITools(req.Tools),
		Stream:              true,
		TopP:                topP,
		MaxCompletionTokens: req.MaxTokens,
	}

	if req.JSONMode {
		openAIReq.ResponseFormat = &openai.ChatCompletionResponseFormat{
			Type: openai.ChatCompletionResponseFormatTypeJSONObject,
		}
	}

	stream, err := o.client.CreateChatCompletionStream(ctx, openAIReq)
	if err != nil {
		var openAIErr *openai.APIError
		if errors.As(err, &openAIErr) {
			return nil, fmt.Errorf("OpenAI API error: %s - %s", openAIErr.Code, openAIErr.Message)
		}
		return nil, fmt.Errorf("stream creation failed: %w", err)
	}

	return newOpenAIStreamWrapper(stream), nil
}
