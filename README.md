# Sensing, Imaging and Signal Processing G2066

Welcome to the 2026 Group Industrial Project GitHub repository for Sensing, Imaging and Signal Processing. This repository will be used to collaborate on code written, and serve as a record of how the project grows over time. 

## Repository Summary
Files:
* `.gitignore`: Used to ignore files or folders from commits, such as propriatary datasets. 
* `.python-version`: Used to define the Python version of the project. The project uses Python 3.13.7
* `README.md`: The file you are reading right now!
* `requirements.txt`: Contains all libraries used in the project. More information on how to use it below. 

Folders that contain a lot of information will themselves have a `README.md` file available. 

## Using the Repository

This repository is set up so that each collaborator works on their own branch, rather than directly on the `main` branch. This helps prevent conflicts and keeps the project organised.

### 1. Cloning the repository

To get a copy of the repository on your computer, run:

    git clone <repository-url>
    cd <repository-folder-name>

When you first clone the repository, you will be on the `main` branch by default.

---

### 2. Viewing all available branches

Each collaborator has their own branch named in the format:

    feature-<name>

For example:

    feature-oscar

To see all branches in the repository, run:

    git branch -a

This will list both local and remote branches.

---

### 3. Switching to your own branch

Once you know the name of your branch, switch to it using:

    git switch feature-<your-name>

Example:

    git switch feature-oscar

After switching, all changes you make will be saved to your own branch, not `main`.

---

### 4. Making and saving changes

You can now edit files, run code, and add new content.

To save your work:

    git add .
    git commit -m "Brief description of your changes"
    git push

---

### 5. Merging your work into `main`

Do not push directly to the `main` branch.

When your work is ready:
1. Push your branch to GitHub
2. Open a Pull Request from your `feature-<name>` branch into `main`
3. Another group member will review and merge your changes

---

### Useful commands

Check which branch you are currently on:

    git branch

View all branches:

    git branch -a

The branch marked with `*` is your current branch.