from crewai import LLM, Agent


class EmailPersonalizationAgents:
    def __init__(self):
        self.llm = LLM(
            # model="groq/llama-3.3-70b-versatile",
            model="groq/deepseek-r1-distill-llama-70b",
            api_key="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",
            temperature=0,
        )

    def personalized_email_agent(self):
        return Agent(
            role="Email Personalization",
            goal="""
            Personalize template emails for recipients using their information.

            Given a template email and recipient information (name, email, bio, last conversation), 
            personalize the email by incorporating the recipient's details 
            into the email while maintaining the core message and structure of the original email. 
            This involves updating the introduction, body, and closing of the email to make 
            it more personal and engaging for each recipient.
            """,
            backstory="""
            As an Email Personalizer, you are responsible for customizing template emails for individual recipients based on their information and previous interactions.
            """,
            verbose=True,
            llm=self.llm,
            max_iter=2,
        )

    def ghostwriter_agent(self):
        return Agent(
            role="Ghostwriter",
            goal="""
                Revise draft emails to adopt the Ghostwriter's writing style.

                Use an informal, engaging, and slightly sales-oriented tone, mirroring the Ghostwriter's final email communication style.
                """,
            backstory="""
                As a Ghostwriter, you are responsible for revising draft emails to match the Ghostwriter's writing style, focusing on clear, direct communication with a friendly and approachable tone.
                """,
            verbose=True,
            llm=self.llm,
            max_iter=2,
        )
