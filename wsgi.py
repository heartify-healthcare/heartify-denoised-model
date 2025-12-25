from app import create_app

# Create the Flask application instance
app = create_app()

if __name__ == "__main__":
    # Exposed port should be 5001 in Development or 5000 in Production
    app.run(host="0.0.0.0", port=5001)
