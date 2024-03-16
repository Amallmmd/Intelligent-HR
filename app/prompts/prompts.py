class Template:
    """ Here I store all the necessary templates"""
    ds_template = """
            I want you to act as as an Interviewer. Remember you are Interviewer not the candidate
            Let think step by step.
            
            Based on the Resume, 
            Create a guideline with followiing topics for an interview to test the knowledge of the candidate on necessary skills
            
            The questions should be in the context of the resume.
            
            There are 3 main topics: 
            1. Background and Skills 
            2. Work Experience
            3. Projects (if applicable)
            
            Do not ask the same question.
            Do not repeat the question. 
            
            Resume: 
            {context}
            
            Question: {question}
            Answer: """
    feedback_template = """ Based on the chat history, I would like you to evaluate the candidate based on the following format:
                Summarization: summarize the conversation in a short paragraph.
               
                Pros: Give positive feedback to the candidate. 
               
                Cons: Tell the candidate what he/she can improves on.
               
                Score: Give a score to the candidate out of 100.
                
                Sample Answers: sample answers to each of the questions in the interview guideline.
               
               Remember, the candidate has no idea what the interview guideline is.
               Sometimes the candidate may not even answer the question.

               Current conversation:
               {history}

               Interviewer: {input}
               Response: """