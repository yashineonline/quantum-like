extensions = ["sphinx_gallery.gen_gallery"]
sphinx_gallery_conf = {
    "examples_dirs": "../examples",  # Path to example scripts
    "gallery_dirs": "auto_examples",  # Output dir for generated pages
    "filename_pattern": r".*",       # Execute all example .py files
}