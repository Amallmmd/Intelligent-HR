class Template:
    """ Here I store all the necessary templates"""
    ds_template = """
            I want you to act as as an Interviewer. Remember you are Interviewer not the candidate
            Let think step by step.
            
            Based on the Resume, 
            Create a guideline with followiing topics for an interview to test the knowledge of the candidate on necessary skills for being a Data Scientist.
            
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