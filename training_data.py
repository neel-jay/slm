# training_data.py

def get_cot_training_data():
    """
    Returns an extensive list of training examples for Chain-of-Thought reasoning.
    Each item includes a question and a detailed, reasoned answer.
    """
    return [
        {
            "question": "I have 5 apples and I buy 3 more boxes of apples. Each box contains 12 apples. How many apples do I have in total?",
            "answer": "Thought: The user starts with 5 apples. They buy 3 more boxes. Each box has 12 apples. First, I need to calculate the total number of new apples from the boxes. That is 3 boxes * 12 apples/box = 36 apples. Then, I need to add this to the initial number of apples. So, 36 new apples + 5 initial apples = 41 apples.\nFinal Answer: You have 41 apples in total."
        },
        {
            "question": "Who was the U.S. President when the first person walked on the moon?",
            "answer": "Thought: The user is asking about the US President at the time of the first moonwalk. First, I need to determine the date of the first moonwalk. The Apollo 11 mission landed on the moon on July 20, 1969. Next, I need to find out who was the US President in July 1969. Richard Nixon was the President from 1969 to 1974. Therefore, Richard Nixon was president.\nFinal Answer: Richard Nixon was the U.S. President when the first person walked on the moon."
        },
        {
            "question": "What is the capital of the country famous for the Eiffel Tower and what is that country's main language?",
            "answer": "Thought: This question has two parts. First, I need to identify the country famous for the Eiffel Tower. The Eiffel Tower is located in Paris, which is in France. So the country is France. Second, I need to find the capital of France, which is Paris. Third, I need to find the main language of France, which is French.\nFinal Answer: The capital is Paris, and the main language is French."
        },
        {
            "question": "A train travels at 60 km/h. How long will it take to travel a distance of 150 km?",
            "answer": "Thought: The user wants to calculate the time it takes for a journey. The formula is Time = Distance / Speed. The distance is 150 km. The speed is 60 km/h. So, Time = 150 km / 60 km/h = 2.5 hours.\nFinal Answer: It will take 2.5 hours to travel a distance of 150 km."
        },
        {
            "question": "Which planet is known as the Red Planet and what is the name of the largest volcano on it?",
            "answer": "Thought: The user is asking two things about a specific planet. First, I need to identify the planet known as the 'Red Planet'. That is Mars. Second, I need to find the name of the largest volcano on Mars. The largest volcano in the solar system is on Mars, and its name is Olympus Mons.\nFinal Answer: The planet known as the Red Planet is Mars, and the largest volcano on it is Olympus Mons."
        },
        {
            "question": "If a shirt costs $20 after a 15% discount, what was the original price?",
            "answer": "Thought: The user wants to find the original price before a discount. The final price ($20) represents 100% - 15% = 85% of the original price. Let the original price be P. So, 0.85 * P = $20. To find P, I need to calculate P = $20 / 0.85. 20 divided by 0.85 is approximately 23.53.\nFinal Answer: The original price of the shirt was approximately $23.53."
        },
        {
            "question": "Combine the mottos of the U.S. Marine Corps and the Royal Air Force.",
            "answer": "Thought: The user wants me to find and combine two mottos. First, find the motto of the U.S. Marine Corps. That is 'Semper Fidelis'. Second, find the motto of the Royal Air Force. That is 'Per ardua ad astra'. Now I will combine them.\nFinal Answer: The combined mottos are 'Semper Fidelis' and 'Per ardua ad astra'."
        }
    ]