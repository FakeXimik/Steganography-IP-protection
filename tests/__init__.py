# NOTE: Python treats folders as packages if you have a __init__.py in them.
#       This is important to avoid path errors when importing our own packages
#       Example:
#           Solved the issue with the line when running pytest:
#               from UUID import *
#
#       Would highly incourage to learn more about this and why you guys should also add __init__.py in the folders you are working on.