# OpenFold Contributor’s Guide

Thank you for taking the time to learn how to contribute to OpenFold\! OpenFold thrives on community contributions; we welcome high quality contributions from everyone that wants to help develop with OpenFold.

## How to make a contribution

To make a contribution, you should first set up your repository, then submit a pull request of the changes you want to make. Below are the steps to setup the openfold3 repository locally.

1. Set up your repository:

   ::::{admonition} Instructions
   1. Make a personal fork of the openfold-3 repository.
   2. Clone openfold-3.
   3. If you already have a fork, make sure to pull to get the latest changes, e.g.

      ```shell
      git fetch origin && git pull origin main
      ```

   4. Install openfold3 locally, using

      ```shell
      pip install .[dev]
      ```

   5. Setup OpenFold3. Make sure to run the full integration tests and that these tests pass. Also run all the unit tests and make sure the unit tests pass.

      ```shell
      $ setup_openfold
      $ pytest openfold3/tests/*
      ```
   ::::

2. Write the changes you want to make. See [\#good-first-issues](https://github.com/aqlaboratory/openfold-3/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22) for some ideas of where to start.  
3. Test your changes

   ::::{admonition} Instructions
   1. Include unit tests to test your changes. If you have limited experience writing tests, we can help. A good starting point is to convert the examples that you used to verify that your changes work into individual test cases
   2. Run all the unit tests, make sure they pass

      ```
      pytest openfold3/tests/*
      ```
   3. If you are adding a new feature, consider adding documentation. It can help make your feature more discoverable.. If you are unsure where to place the documentation, the core team can provide suggestions during review.
   ::::

4. Format the changes. In the OpenFold project, we use Ruff as our Linting tool. You can run Ruff in the same environment with.

    ```shell
    ruff format && ruff check --fix
    ```

5. Open a Pull Request (PR) with your change. See this developer’s guide for making a [PR from your fork of the repo](https://www.contribution-guide.org/#preparing-your-fork).   
     
6. Code review. The OpenFold development team will aim to give feedback that will improve the changes and that matches the codebase style. Please be engaged and respectful in discussing and implementing suggestions from the review.  
     
7. Celebrate. Once the code review is complete, a core developer will merge the PR into openfold3. 

## AI contribution guidelines

We recognize that LLMs and coding agents can provide benefits in terms of speed of code development. 

OpenFold has a small core development team, with finite time for code review and issue responses. As such, we want to ensure that contributions we receive are of high quality and promote the development of the OpenFold community.

To help us conserve our resources, we ask contributors to meet coding standards and engage fully in code reviews. The following guidelines are some steps towards this goal. As the field of AI contribution develops, we may review and edit these guidelines.

- ***All contributions, AI generated or not, are the responsibility of the human contributor.***  
  - Fully understand and review any LLM generated content before submitting PRs / issues. Check if the output makes sense, if the changes work as you anticipated. Provide examples of the test examples that demonstrate your code is working.

- **Issues and Pull Requests should be written by humans.** Issues and Pull Requests are the first line of communication between the contributor and the core development team. We want to fully understand what issue was encountered, and have a dialogue with the contributor about the best path forward.   
  - Exceptions are allowed for translation purposes, or for generating failing test examples, however please disclose the use of tools in this case.

- **“Good First Issues” are reserved for new developers.** Good first issues are chosen as a gateway for new contributors to familiarize themselves with the github pull request workflow and the OpenFold3 codebase by making their first contributions with a simpler issue. Agentic contributions on such issues disrupt this learning process and will be closed.

We reserve the right to close any contributions that we believe do not meet this criteria.