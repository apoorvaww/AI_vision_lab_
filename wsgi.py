"""
wsgi.py
-------
WSGI entrypoint for production application servers (e.g., Gunicorn).
"""

from app import app

if __name__ == "__main__":
    app.run()
