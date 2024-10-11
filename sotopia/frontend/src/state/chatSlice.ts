import { createSlice, PayloadAction } from "@reduxjs/toolkit";

type SliceState = { messages: Message[] };

const initialState: SliceState = {
  messages: [],
};

export const chatSlice = createSlice({
  name: "chat",
  initialState,
  reducers: {
    addUserMessage(
      state,
      action: PayloadAction<{
        content: string;
        imageUrls: string[];
        timestamp: string;
      }>,
    ) {
      const message: Message = {
        sender: "user",
        content: action.payload.content,
        imageUrls: action.payload.imageUrls,
        timestamp: action.payload.timestamp || new Date().toISOString(),
      };
      state.messages.push(message);
    },

    addAssistantMessage(state, action: PayloadAction<string>) {
      console.log("Adding assistant message:", action);
      const message: Message = {
        sender: action.payload.sender,
        content: action.payload.content,
        imageUrls: [],
        timestamp: new Date().toISOString(),
      };
      state.messages.push(message);
    },

    clearMessages(state) {
      state.messages = [];
    },
  },
});

export const { addUserMessage, addAssistantMessage, clearMessages } =
  chatSlice.actions;
export default chatSlice.reducer;
