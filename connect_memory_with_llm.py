def load_llm():
    import streamlit as st
    from huggingface_hub import InferenceClient
    from langchain_core.language_models.llms import LLM
    from typing import Optional, List

    hf_token = st.secrets.get("HF_TOKEN") or os.environ.get("HF_TOKEN")

    class DirectHFLLM(LLM):
        token: str

        @property
        def _llm_type(self):
            return "huggingface"

        def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs):
            client = InferenceClient(
                model="mistralai/Mistral-7B-Instruct-v0.3",
                token=self.token,
                provider="hf-inference"
            )
            response = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512
            )
            return response.choices[0].message.content

    return DirectHFLLM(token=hf_token)
