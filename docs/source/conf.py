# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "OpenFold3"
copyright = "2025, OpenFold Team"
author = "OpenFold Team"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["myst_parser", "sphinxcontrib.mermaid"]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_logo = "../imgs/of-logo.png"
html_favicon = "../imgs/of-logo-small.png"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
myst_enable_extensions = ["colon_fence", "dollarmath", "amsmath"]


# -- Configuration for Mermaid -------------------------------------------------
# Toggle light and dark mode based on Furo's theme and system preferences,
# with a fallback to 'base' (light) if no preference is detected.

mermaid_init_js = """
function initMermaid() {
    const bodyTheme = document.body.dataset.theme;
    let theme = 'base';
    if (bodyTheme === 'dark') theme = 'dark';
    else if (bodyTheme === 'light') theme = 'base';
    else if (window.matchMedia('(prefers-color-scheme: dark)').matches) theme = 'dark';

    mermaid.initialize({
        startOnLoad: false,
        theme: theme
    });
    mermaid.run({ querySelector: '.mermaid' });
}

document.addEventListener('DOMContentLoaded', function() {
    initMermaid();

    new MutationObserver(initMermaid).observe(document.body, {
        attributes: true,
        attributeFilter: ['data-theme']
    });
});
"""
