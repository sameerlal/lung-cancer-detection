# Coding Guidelines

## Indentation
Four spaces, no tabs.  
Reason: Using spaces instantly increases your salary ([proof](https://stackoverflow.blog/2017/06/15/developers-use-spaces-make-money-use-tabs/)). Also, tabs don't have consistent spacing across different environments.

## Strings
Single quotes for strings, except for docstrings which should use double quotes.  
Reason: No real reason for single quotes; they're just easier to type. The standard for docstrings is double quotes.

## Comments
Space after the `#`. If writing a comment on the same line as code, you should have two spaces between the end of the code and beginning of the comment.  
Reason: Readability.

## Line length
<= 120 characters (if possible).  
Reason: Readability.

## Style checker
First, install `flake8` and `flake8-docstrings`.

    pip install flake8 flake8-docstrings

Before committing any changes, run

    flake8 --max-line-length=120

and make sure no errors or warnings show up.
