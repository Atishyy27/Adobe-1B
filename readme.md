## Features

-   **Multi-File Upload**: Users can upload 3-10 PDF documents for analysis.
-   **Persona-Based Analysis**: Users define a persona and a job-to-be-done to guide the analysis.
-   **Interactive PDF Viewer**: Utilizes the Adobe PDF Embed API for a seamless and powerful PDF viewing experience.
-   **Dynamic Insights Panel**: Displays the structured outline and persona-relevant sections extracted by the backend.
-   **Click-to-Navigate**: Clicking on an insight in the side panel instantly navigates to the relevant page in the correct document.

## Project Structure
```
├── backend/
│   ├── main.py           # FastAPI application
│   ├── Dockerfile        # Docker configuration for the backend
│   └── requirements.txt  # Python dependencies
├── frontend/
│   ├── index.html        # Main application page
│   ├── script.js         # Frontend logic and API application
├── docker-compose.yml    # Docker Compose to run both services
└── README.md             # This file
```
