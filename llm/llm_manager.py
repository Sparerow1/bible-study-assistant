from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from config.config_setup_class import BibleQAConfig
from typing import Optional, Dict, Any

class LLMManager:
    """Manages the Language Model and conversation chain."""
    
    def __init__(self, config: BibleQAConfig):
        self.config = config
        self.llm = None
        self.memory = None
        self.qa_chain = None
        
    def initialize_llm(self, google_api_key: str) -> bool:
        """Initialize Gemini LLM."""
        try:
            print("🤖 Initializing Gemini LLM...")
            self.llm = ChatGoogleGenerativeAI(
                google_api_key=google_api_key,
                streaming=True,
                convert_system_message_to_human=True,
                model=self.config.llm_model,
                temperature=self.config.llm_temperature,
                callbacks=[StreamingStdOutCallbackHandler()]
            )
            
            # Test the LLM
            test_response = self.llm.invoke("Say 'Hello' in Chinese")
            print(f"✅ LLM initialized! Test response: {test_response.content[:50]}...")
            return True
            
        except Exception as e:
            print(f"❌ LLM initialization failed: {e}")
            print("💡 Please check your GOOGLE_API_KEY and model availability.")
            return False
    
    def initialize_memory(self) -> bool:
        """Initialize conversation memory."""
        try:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            print("✅ Memory initialized")
            return True
        except Exception as e:
            print(f"❌ Error initializing memory: {e}")
            return False
    
    def create_qa_chain(self, retriever) -> bool:
        """Create the conversational retrieval chain."""
        try:
            custom_prompt = self._create_pastoral_prompt()
            
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=self.memory,
                verbose=False,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": custom_prompt},
            )
            print("✅ Q&A chain created successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error creating Q&A chain: {e}")
            return False
    
    def _create_pastoral_prompt(self) -> PromptTemplate:
        """Create the pastoral guidance prompt template."""
        return PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template="""你是一位博学、幽默、且富有同情心的圣经学者和属灵导师，同时也是一位经验丰富的圣经学习助手。你的角色是提供全面、深思熟虑且具有属灵启发的圣经和基督教信仰答案，并关怀和指导询问者。

请以“圣经学习助手”自称，并在回答中体现牧者的关怀和智慧。你的回答应当帮助人们更好地理解神的话语，并在属灵上成长。

作为一位好圣经学习助手，请遵循以下原则：

圣经学习助手品格与策略：
• 以基督的爱心和耐心回应每一个问题
• 倾听并理解询问者内心的需要和挣扎
• 提供既有真理又有恩典的平衡回答
• 用温柔而坚定的语气传达神的话语
• 避免论断，而是以慈爱引导人回到神面前
• 为询问者的属灵成长和实际需要祷告考虑
• 提供既有深度又容易理解的解释

请使用以下圣经经文和背景来回答问题，且不要超出以下经文和背景所提供的范围回答，如果不知道，就回答不知道。请提供详细且结构清晰的回答，包括：

1. 对问题的直接回答（以牧者的关怀开始）
2. 相关的圣经经文和引用（解释其中的属灵含义）
3. 历史和文化背景（如适用）
4. 实际应用或属灵见解（针对现代信徒的生活）
5. 与相关圣经概念的交叉引用
6. 鼓励和祷告方向（如适用）

圣经背景：
{context}

之前的对话：
{chat_history}

问题：{question}

作为一位圣经学习助手，请提供一个全面的答案，对于那些寻求更好理解神话语的人有帮助。请包含具体的经文引用，并以清晰易懂的方式解释含义。在回答中体现关怀、智慧和同情心是圣经学习助手的品格。

请以提问者所用的语言回答问题，并确保回答符合圣经的教导和原则。

如果遇到与圣经不相关的问题，请温柔地将话题引导回到圣经相关的问题，同时关心询问者的属灵需要。避免使用个人观点或未经证实的解释，但要以圣经学习助手的经验和智慧来应用圣经真理。

记住，你不仅是在传授知识，更是在牧养灵魂，帮助人们在基督里成长。

回复："""
        )
    
    def process_question(self, question: str) -> Optional[Dict[str, Any]]:
        """Process a question through the QA chain."""
        if not self.qa_chain:
            print("❌ QA chain not initialized!")
            return None
            
        try:
            result = self.qa_chain({"question": question})
            return result
        except Exception as e:
            print(f"❌ Error processing question: {e}")
            return None
    
    def clear_memory(self) -> bool:
        """Clear conversation memory."""
        try:
            if self.memory:
                self.memory.clear()
                return True
            return False
        except Exception as e:
            print(f"❌ Error clearing memory: {e}")
            return False