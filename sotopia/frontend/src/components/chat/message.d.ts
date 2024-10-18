type Message = {
  sender: "user" | "assistant" | "agent-1" | "agent-2";
  content: string;
  imageUrls: string[];
  timestamp: string;
};
