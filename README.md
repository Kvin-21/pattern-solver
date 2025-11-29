# Pattern Solver

A web app for identifying and solving number sequences. Enter a sequence like `1, 2, 3, 4` and it'll tell you the pattern and predict the next terms.

## Features

- **Pattern Solver**: Detects arithmetic, geometric, Fibonacci, polynomial, and many other sequence types
- **Pattern Game**: Test your pattern recognition skills with increasingly difficult challenges
- **OEIS Integration**: Searches 300k+ sequences from the Online Encyclopedia of Integer Sequences
- **Blog & Library**: Educational content about mathematical patterns

## Running Locally

```bash
pip install -r requirements.txt
python app/app.py
```

The app runs on `http://localhost:5000`.

## Environment Variables

Create a `.env` file:

```
MONGODB_URI=your_mongodb_connection_string
SECRET_KEY=your_secret_key
```

## Dependencies

- Flask
- MongoDB (via pymongo)
- NumPy & scikit-learn for pattern analysis
- bcrypt for password hashing