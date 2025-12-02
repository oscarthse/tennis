import streamlit as st
import pandas as pd
from src.dashboard.components.navigation import sidebar_navigation
from src.dashboard.components.mermaid import render_mermaid

st.set_page_config(page_title="Docker Deep Dive", page_icon="🐳", layout="wide")
sidebar_navigation()

st.title("🐳 Docker Deep Dive: From Text to Process")

st.markdown("""
> **"It works on my machine"** is the most expensive phrase in software engineering.
> Docker fixes this by shipping the **machine** along with the code.
""")

# --- SECTION 1: THE GOLDEN TRIANGLE ---
st.header("1. The Golden Triangle △")
st.markdown("To understand Docker, you must understand the relationship between these three things:")

col1, col2, col3 = st.columns(3)
with col1:
    st.info("**1. Dockerfile** (The Recipe)")
    st.markdown("A text file with instructions. It's just code.")
with col2:
    st.warning("**2. Image** (The Frozen Meal)")
    st.markdown("The result of 'building' the Dockerfile. It is **Read-Only** and immutable.")
with col3:
    st.success("**3. Container** (The Hot Meal)")
    st.markdown("A running instance of an Image. It is **Alive** and has a writable layer.")

render_mermaid("""
graph LR
    DF[("Dockerfile <br> (Text)")] -->|docker build| IMG[("Image <br> (Binary)")]
    IMG -->|docker run| CONT[("Container <br> (Process)")]

    style DF fill:#e3f2fd
    style IMG fill:#fff3e0
    style CONT fill:#e8f5e9
""", height=200)

st.markdown("---")

# --- SECTION 2: THE DOCKERFILE ---
st.header("2. The Dockerfile: Writing the Recipe 📝")
st.markdown("A `Dockerfile` tells Docker how to build your application from scratch. Let's analyze a real one.")

col_code, col_explain = st.columns([1, 1])

with col_code:
    st.code("""
    # 1. Base Image (The Foundation)
    FROM python:3.9-slim

    # 2. Working Directory (The Kitchen)
    WORKDIR /app

    # 3. Dependencies (The Ingredients)
    COPY requirements.txt .
    RUN pip install -r requirements.txt

    # 4. Source Code (The Meal)
    COPY . .

    # 5. Command (Serve it)
    CMD ["python", "app.py"]
    """, language="dockerfile")

with col_explain:
    st.markdown("""
    *   **FROM**: We don't start from zero. We start with a pre-built Linux OS that already has Python installed (`python:3.9-slim`).
    *   **WORKDIR**: Creates a folder `/app` inside the image and moves us there. Like `cd /app`.
    *   **COPY**: Copies files from your **Laptop** (Host) to the **Image**.
    *   **RUN**: Runs a command *during the build process*. This installs libraries into the image.
    *   **CMD**: The default command to run when the container *starts*.
    """)

st.markdown("### 💡 Best Practice: Layer Caching")
st.markdown("""
Notice we copied `requirements.txt` **before** the rest of the code?
*   Docker caches every step.
*   If you change `app.py`, Docker sees that `requirements.txt` hasn't changed.
*   It **skips** the slow `pip install` step and uses the cache.
*   If you `COPY . .` first, any code change breaks the cache for everything following it.
""")

st.markdown("---")

# --- SECTION 3: THE IMAGE ---
st.header("3. The Image: The Onion 🧅")
st.markdown("An Image is not a single file. It is a stack of **Read-Only Layers**.")

render_mermaid("""
graph BT
    L1["Layer 1: Ubuntu Core (100MB)"]
    L2["Layer 2: Python Files (50MB)"]
    L3["Layer 3: Pip Packages (200MB)"]
    L4["Layer 4: Your App Code (5KB)"]

    L2 --> L1
    L3 --> L2
    L4 --> L3

    style L1 fill:#f5f5f5
    style L2 fill:#f5f5f5
    style L3 fill:#f5f5f5
    style L4 fill:#f5f5f5
""", height=300)

st.markdown("""
*   **Union File System**: Docker merges these layers into a single view.
*   **Immutability**: Once built, these layers **NEVER** change. This guarantees that if it works on my machine, it works on yours.
*   **Sharing**: If you have 10 images based on `python:3.9`, the 100MB base layer is stored **only once** on your disk.
""")

st.markdown("---")

# --- SECTION 4: THE CONTAINER ---
st.header("4. The Container: Coming Alive 🧟")
st.markdown("When you run an image, Docker adds a thin **Read-Write Layer** on top.")

render_mermaid("""
graph BT
    subgraph Image ["Read-Only Image"]
        L1["Ubuntu"]
        L2["Python"]
        L3["App Code"]
    end

    subgraph Container ["Running Container"]
        RW["Writable Layer <br> (Logs, Temp Files)"]
    end

    RW --> L3
    L3 --> L2
    L2 --> L1

    style RW fill:#ffccbc
""", height=350)

st.markdown("""
*   **Ephemeral**: If you delete the container, the Writable Layer is **destroyed**. All your logs and temp files are gone.
*   **Copy-on-Write**: If the container wants to modify a file from the Image (e.g., `app.py`), Docker first **copies** it up to the Writable Layer. The original Image remains untouched.
""")

st.markdown("---")

# --- SECTION 5: COMMAND CHEATSHEET ---
st.header("5. The Toolbelt: Essential Commands 🛠️")

commands = [
    {"Command": "docker build -t my-app .", "Action": "Bake", "Description": "Turn Dockerfile into an Image named 'my-app'."},
    {"Command": "docker run -p 80:80 my-app", "Action": "Serve", "Description": "Start a Container from the Image. Map port 80."},
    {"Command": "docker ps", "Action": "Check", "Description": "List currently running containers."},
    {"Command": "docker stop <id>", "Action": "Halt", "Description": "Gracefully stop a container."},
    {"Command": "docker rm <id>", "Action": "Clean", "Description": "Delete a stopped container (removes Writable Layer)."},
    {"Command": "docker rmi <image>", "Action": "Destroy", "Description": "Delete an Image from your disk."},
    {"Command": "docker exec -it <id> bash", "Action": "Enter", "Description": "Open a shell INSIDE the running container."}
]

st.table(pd.DataFrame(commands))

st.markdown("---")

# --- SECTION 6: INTERACTIVE SIMULATION ---
st.header("6. Simulation: The Build Process 🏗️")

if st.button("Simulate 'docker build'"):
    with st.status("Building Image...", expanded=True) as status:
        st.write("Step 1/5 : FROM python:3.9-slim")
        st.write(" ---> Using cache (Layer A)")

        st.write("Step 2/5 : WORKDIR /app")
        st.write(" ---> Using cache (Layer B)")

        st.write("Step 3/5 : COPY requirements.txt .")
        st.write(" ---> Using cache (Layer C)")

        st.write("Step 4/5 : RUN pip install -r requirements.txt")
        st.write(" ---> Running in 1a2b3c...")
        st.write(" ---> Installing pandas, numpy...")
        st.write(" ---> Removing intermediate container")
        st.write(" ---> Created Layer D")

        st.write("Step 5/5 : COPY . .")
        st.write(" ---> Created Layer E")

        status.update(label="Build Complete!", state="complete", expanded=True)

    st.success("Successfully built image: `tennis-app:latest`")

st.markdown("---")

# --- SECTION 7: EXERCISES ---
st.header("7. Exercises 📝")
st.info("""
1.  **Inspect**: Run `docker history python:3.9-slim`. You will see the layers!
2.  **Break Cache**: Change a comment in your `Dockerfile` at the top. Run build again. See how it re-runs everything below it?
3.  **Enter**: Run a container and use `docker exec -it <id> /bin/bash`. Look around. It's a whole new OS inside.
""")
