import os
from dotenv import load_dotenv
from pymongo import MongoClient
from flask import Flask, render_template, request, session, redirect, url_for, flash, send_from_directory
import bcrypt
import numpy as np
from math import sqrt, pow
import re
from bson.objectid import ObjectId
import datetime
from flask_sitemap import Sitemap
from flask_compress import Compress

# Load environment variables
load_dotenv()

# MongoDB configuration
MONGODB_URI = os.getenv('MONGODB_URI')
SECRET_KEY = os.getenv('SECRET_KEY')

# Initialize MongoDB client and database
client = MongoClient(MONGODB_URI)
db = client['PatternSolverDB']
users_collection = db['users']
patterns_collection = db['patterns']

# Flask app initialization
app = Flask(__name__)
app.secret_key = SECRET_KEY

ext = Sitemap(app)

@ext.register_generator
def index():
    yield 'home', {}
    yield 'login', {}
    yield 'register', {}
    yield 'game', {}
    yield 'solver', {}
    yield 'settings', {}

Compress(app)

# Password hashing functions
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())


def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

def format_number(num):
    """Format number to remove .0 if it's a whole number"""
    if isinstance(num, int):
        return num
    if isinstance(num, float):
        if num.is_integer():
            return int(num)
        return round(num, 2)
    return num

def linear(x):
    return x

def square(x):
    return x**2

def cube(x):
    return x**3

def exponential(x):
    return 2**x

def square_root(x):
    return sqrt(x)

def quadratic(x):
    return x**2 + x

def cubic_minus(x):
    return x**3 - x**2

def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for _ in range(2, n+1):
            a, b = b, a + b
        return b

def is_fibonacci(numbers):
    if len(numbers) < 3:
        return False
    return all(abs(numbers[i] - (numbers[i-1] + numbers[i-2])) < 0.1 
              for i in range(2, len(numbers)))

def next_fibonacci_terms(numbers, count=3):
    sequence = list(numbers)
    for _ in range(count):
        next_num = sequence[-1] + sequence[-2]
        sequence.append(next_num)
    return [round(x, 2) for x in sequence[-count:]]

def prod(lst):
    result = 1
    for x in lst:
        result *= x
    return result

def check_geometric_varying(numbers):
    """Check if sequence follows a geometric pattern with varying ratio"""
    if len(numbers) < 3:
        return False, None
    
    ratios = [numbers[i]/numbers[i-1] for i in range(1, len(numbers))]
    ratio_diffs = [ratios[i] - ratios[i-1] for i in range(1, len(ratios))]
    
    # Check if ratio differences are consistent
    if len(set(round(diff, 6) for diff in ratio_diffs)) == 1:
        ratio_increment = ratio_diffs[0]
        next_ratio = ratios[-1] + ratio_increment
        return True, lambda n: numbers[-1] * next_ratio if n == len(numbers) + 1 else \
                             numbers[-1] * prod([ratios[-1] + (ratio_increment * i) for i in range(1, n - len(numbers) + 1)])
    return False, None

def get_pattern_explanation(pattern_type):
    explanations = {
        "Linear": "Linear pattern: Each number increases by 1",
        "Square": "Square numbers: Each number is n²",
        "Cube": "Cube numbers: Each number is n³",
        "Exponential": "Powers of 2: Each number is 2ⁿ",
        "Square Root": "Square root pattern: Each number is √n",
        "Quadratic": "Quadratic pattern: Each number is n² + n",
        "Cubic": "Cubic pattern: Each number is n³ - n²",
        "Fibonacci": "Fibonacci sequence: Each number is the sum of the two preceding numbers"
    }
    return explanations.get(pattern_type, "Complex pattern: Try to spot the mathematical relationship")

def identify_pattern_type(numbers):
    # First check 
    
    n = len(numbers)
    x = list(range(1, n + 1))
    
    # Check for Fibonacci sequence
    if is_fibonacci(numbers):
        next_terms = next_fibonacci_terms(numbers)
        return "Fibonacci sequence: each number is the sum of the two preceding numbers", \
               lambda n: next_fibonacci_terms(numbers, 1)[0] if n == len(numbers) + 1 else \
                        next_fibonacci_terms(numbers, n - len(numbers))[-1] if n > len(numbers) else \
                        numbers[n-1]
    
    # Check for arithmetic sequence
    if len(set(np.diff(numbers))) == 1:
        d = numbers[1] - numbers[0]
        return f"arithmetic sequence: starts at {numbers[0]}, increases by {d}", \
               lambda n: numbers[0] + (n-1)*d
    
    # Check for geometric sequence
    if len(numbers) > 1:
        ratios = [numbers[i]/numbers[i-1] for i in range(1, len(numbers))]
        if len(set([round(r, 6) for r in ratios])) == 1:
            r = ratios[0]
            return f"geometric sequence: starts at {numbers[0]}, multiplies by {round(r,2)}", \
                   lambda n: numbers[0] * pow(r, n-1)
    
    # Check for square numbers
    if all(abs(i*i - num) < 0.1 for i, num in zip(range(1, n + 1), numbers)):
        return "square numbers: n²", square
    
    # Check for cube numbers
    if all(abs(i**3 - num) < 0.1 for i, num in zip(range(1, n + 1), numbers)):
        return "cube numbers: n³", cube
    
    # Check for square roots
    if all(abs(sqrt(i) - num) < 0.1 for i, num in zip(range(1, n + 1), numbers)):
        return "square roots: √n", square_root
    
    # Check for triangular numbers
    if all(abs((i*(i+1)/2) - num) < 0.1 for i, num in zip(range(1, n + 1), numbers)):
        return "triangular numbers: n(n+1)/2", lambda n: n*(n+1)/2

    is_varying_geometric, varying_formula = check_geometric_varying(numbers)
    if is_varying_geometric:
        ratios = [round(numbers[i]/numbers[i-1], 2) for i in range(1, len(numbers))]
        pattern_desc = f"geometric sequence with varying ratio (ratios: {ratios})"
        return pattern_desc, varying_formula

    
    # If no pattern is identified, use polynomial fitting
    coeffs = np.polyfit(x, numbers, min(n-1, 3))
    poly = np.poly1d(coeffs)
    
    # Only return polynomial if it fits well
    residuals = [abs(poly(i+1) - num) for i, num in enumerate(numbers)]
    if max(residuals) < 0.1:
        return f"polynomial: degree {len(coeffs)-1}", poly
    
    return "unknown pattern", lambda n: None

def generate_pattern(level):
    patterns = [
        (linear, "Linear"),
        (square, "Square"),
        (cube, "Cube"),
        (exponential, "Exponential"),
        (square_root, "Square Root"),
        (quadratic, "Quadratic"),
        (cubic_minus, "Cubic"),
        (fibonacci, "Fibonacci")
    ]
    
    pattern_func, pattern_name = patterns[min(level-1, len(patterns)-1)]
    length = np.random.randint(4, 7)
    
    # Generate random starting point between 1 and 10
    start = np.random.randint(1, 11)
    
    # Decide if sequence should be ascending or descending
    is_ascending = np.random.choice([True, False])
    
    if is_ascending:
        sequence = [format_number(float(pattern_func(i))) for i in range(start, start + length)]
        next_term = format_number(float(pattern_func(start + length)))
    else:
        sequence = [format_number(float(pattern_func(i))) for i in range(start + length - 1, start - 1, -1)]
        next_term = format_number(float(pattern_func(start - 1)))
    
    # Ensure next term is positive
    if next_term <= 0:
        return generate_pattern(level)  # Recursively try again
        
    explanation = get_pattern_explanation(pattern_name)
    return sequence, next_term, explanation, pattern_name

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if users_collection.find_one({'username': username}):
            return render_template('register.html', error="Username already taken")
        
        hashed_password = hash_password(password)
        users_collection.insert_one({
            'username': username,
            'password': hashed_password,
            'score': 0,
            'level': 1,
            'patterns': []
        })
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = users_collection.find_one({'username': username})
        if user and check_password(password, user['password']):
            session['user_id'] = str(user['_id'])  # Use MongoDB’s `_id`
            return redirect(url_for('home'))
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('home'))

@app.route('/')
def home():
    user = None
    if 'user_id' in session:
        user = users_collection.find_one({'_id': ObjectId(session['user_id'])})
    return render_template('home.html', user=user)

@app.route('/settings')
def settings():
    user = None
    patterns = None
    if 'user_id' in session:
        user = users_collection.find_one({'_id': ObjectId(session['user_id'])})
        patterns = list(patterns_collection.find({'user_id': ObjectId(session['user_id'])}).sort('timestamp', -1).limit(10))
    return render_template('settings.html', user=user, patterns=patterns)

@app.route('/game', methods=['GET', 'POST'])
def game():
    user = None
    if 'user_id' in session:
        user = users_collection.find_one({'_id': ObjectId(session['user_id'])})
    
    if request.method == 'POST':
        user_guess = request.form.get('guess')
        if not user_guess:
            return render_template('game.html', pattern=session['pattern'], error="Please enter a guess.", user=user)
        
        try:
            user_guess = float(user_guess)
        except ValueError:
            return render_template('game.html', pattern=session['pattern'], error="Invalid guess format.", user=user)
        
        if abs(user_guess - session['correct_answer']) < 0.1:
            if user:
                new_score = user.get('score', 0) + 10
                old_level = user.get('level', 1)
                new_level = min(new_score // 50 + 1, 8)
                update_data = {'score': new_score}
                if new_level > old_level:
                    update_data['level'] = new_level
                    update_data['level_score'] = new_score
                users_collection.update_one({'_id': ObjectId(session['user_id'])}, {'$set': update_data})


            
            session['tries'] = 0  # Reset tries after correct answer
            # Generate new pattern for next round
            session['pattern'], session['correct_answer'], session['explanation'], session['pattern_name'] = generate_pattern(user.get('level', 1) if user else 1)

            return render_template('game.html', pattern=session['pattern'], 
                                success=f"Correct! Score: {user.get('score', 0) if user else 'Not logged in'}", user=user)
        else:
            session['tries'] += 1
            if session['tries'] >= 3:
                session['tries'] = 0
                if user:
                    user['score'] = user.get('level_score', 0)  # Reset score to level start
                    
                
                # Store current pattern info for error message
                current_level = user.get('level', 1) if user else 1
                correct_answer = session['correct_answer']
                explanation = session['explanation']
                pattern_name = session['pattern_name']
                
                # Generate new pattern immediately
                session['pattern'], session['correct_answer'], session['explanation'], session['pattern_name'] = generate_pattern(current_level)
                
                # Use the stored pattern info for the error message
                return render_template('game.html', 
                                    pattern=session['pattern'], 
                                    error=f"Level reset! The answer was {correct_answer}. This was a {pattern_name} pattern: {explanation}",
                                    user=user)
            return render_template('game.html', pattern=session['pattern'], 
                                error=f"Wrong! Tries left: {3-session['tries']}", user=user)
    
    # Initial pattern generation or page refresh
    current_level = user.get('level', 1) if user else 1
    session['pattern'], session['correct_answer'], session['explanation'], session['pattern_name'] = generate_pattern(current_level)
    session['tries'] = 0
    return render_template('game.html', pattern=session['pattern'], user=user)

@app.route('/solver', methods=['GET', 'POST'])
def solver():
    user = None
    if 'user_id' in session:
        user = users_collection.find_one({'_id': ObjectId(session['user_id'])})
    
    if request.method == 'POST':
        pattern_str = request.form.get('pattern')
        try:
            numbers = [float(x.strip()) for x in pattern_str.split(',')]
            if len(numbers) < 3:
                return render_template('solver.html', error="Please enter at least 3 numbers", user=user)
            
            pattern_type, formula = identify_pattern_type(numbers)
            
            if formula is None:
                return render_template('solver.html', 
                                    error="Unable to identify a clear pattern in the sequence",
                                    original_pattern=[format_number(x) for x in numbers],
                                    user=user)
            
            try:
                if callable(formula):
                    next_terms = [float(formula(len(numbers) + i)) for i in range(1, 4)]
                else:  # numpy poly1d object
                    next_terms = [float(formula(len(numbers) + i)) for i in range(1, 4)]
                
                # Format the numbers
                next_terms = [format_number(x) for x in next_terms]
                formatted_original = [format_number(x) for x in numbers]
                
                if user:
                            patterns_collection.insert_one({
                                'user_id': ObjectId(session['user_id']) if user else None,
                                'sequence': pattern_str,
                                'solution': f"Type: {pattern_type}, Next terms: {next_terms}",  # Missing comma here
                                'timestamp': datetime.datetime.now()
                            })



                
                return render_template('solver.html', 
                                    pattern_type=pattern_type,
                                    next_terms=next_terms,
                                    original_pattern=formatted_original,
                                    user=user)
            except (TypeError, ValueError):
                return render_template('solver.html', 
                                    error="Unable to calculate next terms for this pattern",
                                    original_pattern=[format_number(x) for x in numbers],
                                    user=user)
                                
        except (ValueError, ZeroDivisionError) as e:
            return render_template('solver.html', error=f"Invalid pattern format: {str(e)}", user=user)
            
    return render_template('solver.html', user=user)


@app.route('/google925b0ffc084ab8b1.html')
def serve_verification_file():
    return "google-site-verification: google925b0ffc084ab8b1.html"

@app.route('/sitemap.xml')
def serve_sitemap():
    return send_from_directory('static', 'sitemap.xml')

@app.route('/ads.txt')
def serve_ads_file():
    return send_from_directory(
        'static',
        'Ads.txt',
        mimetype='text/plain'
    )

@app.route('/privacy-policy')
def serve_pp_file():
    return send_from_directory(
        'static',
        'privacy-policy.txt',
        mimetype='text/plain'
    )

@app.route('/robots.txt')
def serve_rob():
    return "Sitemap: https://pattern-solver-app.azurewebsites.net/sitemap.xml"

@app.route('/.well-known/pki-validation/7D3D3BA0B414B564BF03E299F5FCB2D3.txt')
def serve_val():
    return send_from_directory(
        'static',
        '7D3D3BA0B414B564BF03E299F5FCB2D3.txt',
        mimetype='text/plain'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=false)
