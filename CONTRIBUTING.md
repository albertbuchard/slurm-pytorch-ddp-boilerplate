# Contributing to `slurm-pytorch-ddp-boilerplate`

First off, thank you for considering contributing to `slurm-pytorch-ddp-boilerplate`. It's people like you that make this project such a great tool.

## Table of Contents
 
1. [Getting Started](#getting-started)
2. [How to Contribute](#how-to-contribute)
3. [Branching Strategy](#branching-strategy)
4. [Pull Request Process](#pull-request-process)
5. [Issue Labels](#issue-labels)
6. [Additional Notes](#additional-notes)

--- 

## Getting Started

- **Fork** the repository to your GitHub account.
- **Clone** your forked repository and navigate to the directory:
  ```bash
  git clone https://github.com/YOUR_USERNAME/slurm-pytorch-ddp-boilerplate.git
  cd slurm-pytorch-ddp-boilerplate
  ```
- **Add** the main repository as a remote to keep in sync:
    ```bash
    git remote add upstream https://github.com/albertbuchard/slurm-pytorch-ddp-boilerplate.git
    ```

- **Synchronize** your fork with the main repository:
    ```bash
  git fetch upstream
  git checkout development
  git merge upstream/development
  ```

- **Propose or take on an issue** to work on. If you're unsure where to start, check out the [issues](https://github.com/albertbuchard/slurm-pytorch-ddp-boilerplate/issues) section.
- Always work on a **new branch** for your changes, referencing the issue number: 
  ```bash
  git checkout -b issue-#-descriptive-title development
  ```

- Make your changes, **commit** them, and **push** to your fork:
  ```bash
  git add .
  git commit -m "Descriptive commit message about your changes"
  git push origin issue-#-descriptive-title
  ```

- Finally, navigate to your fork on GitHub and click the `New pull request` button to start the process of creating a PR to the `development` branch of the main repository.

- Be sure to follow the [Pull Request Process](#pull-request-process) below.
 
---

## How to Contribute

1. **Feature Proposals**:
   - For proposing a new feature, use the **Feature Proposal Template** available in the issues section.
   - Fill in the template with as much detail as possible. The clearer and more comprehensive your proposal, the faster we can collaborate and discuss.

2. **Bug Reports**:
   - If you've identified a bug, use the **Bug Report Template** to provide information about the problem.
   - Clearly describe the issue, including steps to reproduce and any relevant details about your configuration.

3. **Improving Documentation**:
   - Documentation is crucial! If you see an area that could use improvement or lacks clarity, please propose your changes.

4. **Reviewing Pull Requests**:
   - If you're looking to provide feedback, consider reviewing open pull requests. Your insights and review comments can guide the project to even higher standards.

---

## Branching Strategy

- The `master` branch is the main branch where the source code reflects a production-ready state.
- The `development` branch is the main branch for ongoing development. Contributors should always create a new branch off of `development` and submit pull requests back into this branch.
- Periodically, the changes from the `development` branch will be merged into `master` and tagged with a release number.

---

## Pull Request Process

1. **Submit to `development`**: Always direct your pull requests to the `development` branch, not `master`.
   
2. **Title Format**: Please format your PR titles as `#IssueNumber Descriptive Title`. 
   - Example: `#23 Add user authentication support`

3. **Description**: In the PR description, provide a detailed overview of the changes, the motivation behind them, and any additional context that might be useful for reviewers.

4. **Link Related Issues**: If your PR resolves any open issues, mention them in the description to create a link between the PR and issue.

5. **Review**: Once your PR is submitted, it'll be reviewed by the maintainers. Address any feedback or changes requested.

6. **Merging**: Once your PR is approved, it will be merged into the `development` branch. Periodic merges from `development` to `master` represent new versions of the project.

---

## Issue Labels

The `bug` and `feature` labels are the main ones used to categorize issues and pull requests.
Other labels provide additional information about the issue or pull request.
The `question` label should not be used on its own, as questions should be asked in the discussion forum. 

---

## Additional Notes

- Ensure your code adheres to the existing style for consistency.
- Before submitting a large PR, consider opening an issue to discuss the changes and get feedback.
- Always aim for clear, concise, and well-commented code. This makes it easier for others to understand and build upon your contributions.

---

Thank you for your contributions!