"""
🧠 Neuro Brain — Python Logic Lesson
This script demonstrates exactly how the Python logic you just learned
(variables, loops, if-statements, functions) is used to make this AI "Brain" learn.

Run this script to see the logic in action:
    python python_brain_lesson.py
"""

import time

# =====================================================================
# 1. VARIABLES (Storing our AI's knowledge)
# =====================================================================
# The "Brain" starts with a random, wrong guess about how to solve a problem.
# In a real neural network, this is called a "Weight".
brain_weight = 0.5 

# Our goal: We want the brain to turn the input (2) into the target output (10)
# The correct math is 2 * 5 = 10. So the brain needs to learn that the weight should be 5.
input_data = 2.0
target_output = 10.0

# How fast the brain learns from its mistakes
learning_rate = 0.1  


# =====================================================================
# 2. FUNCTIONS (Building reusable blocks of logic)
# =====================================================================
def make_prediction(current_weight, data):
    """The brain multiplies its current knowledge (weight) by the input."""
    return current_weight * data

def calculate_error(prediction, target):
    """How far off was the brain's guess from the correct answer?"""
    return prediction - target


# =====================================================================
# 3. LOOPS & CONDITIONAL LOGIC (The actual "Learning" process)
# =====================================================================
print("=" * 50)
print("🧠 STARTING BRAIN TRAINING LOOP")
print("=" * 50)
print(f"Goal: When I input {input_data}, the brain should output {target_output}.")
print(f"Initial Brain Weight (Knowledge): {brain_weight}\n")

# Use a FOR loop to let the brain practice 15 times (called "Epochs" in AI)
for epoch in range(1, 16):
    
    # Step A: The brain makes a guess
    current_guess = make_prediction(brain_weight, input_data)
    
    # Step B: We check how wrong the guess was
    error = calculate_error(current_guess, target_output)
    
    # Print the current state
    print(f"Practice Round {epoch:02d} | Guess: {current_guess:05.2f} | Error: {abs(error):05.2f}", end=" | ")
    
    # IF/ELSE LOGIC (If the error is tiny, the brain has learned enough!)
    if abs(error) < 0.1:
        print("✅ The Brain has learned the pattern!")
        break  # Exit the loop early
    else:
        print("❌ Still learning...")
        
    # Step C: Update the brain's knowledge based on the error
    # If the guess was too low, the weight goes up. If too high, it goes down.
    # This single line of math is the foundation of ALL Machine Learning (Gradient Descent).
    brain_weight = brain_weight - (error * input_data * learning_rate)
    
    time.sleep(0.3) # Pause so you can watch it learn

print("\n" + "=" * 50)
print("🎯 TRAINING COMPLETE")
print("=" * 50)
print(f"Final Brain Weight: {brain_weight:.2f} (It learned the correct multiplier!)")

# 4. EXCEPTION HANDLING (Testing the trained brain safely)
print("\nLet's test the trained brain with new data!")
try:
    new_input = 5.0
    final_prediction = make_prediction(brain_weight, new_input)
    print(f"Inputting {new_input} -> Brain predicts: {final_prediction:.2f}")
    print(f"(Math check: 5 * 5 = 25. The AI is correct!)")
except Exception as e:
    print(f"[!] Something went wrong during testing: {e}")
