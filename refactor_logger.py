import os
import glob
import re

files = glob.glob('/media/dbosmrt/Data1/Ma-at-/**/*.py', recursive=True)

for filepath in files:
    if filepath.endswith('logger.py'):
        continue
    with open(filepath, 'r') as f:
        content = f.read()

    if 'get_logger(__name__)' in content:
        continue
        
    original = content
    
    if 'logging.getLogger' in content:
        # replace getLogger(__name__) and getLogger("...")
        content = re.sub(r'logger\s*=\s*logging\.getLogger\(__name__\)', 'logger = get_logger(__name__)', content)
        
        if 'import logging' in content and 'from agent.utils.logger import get_logger' not in content:
            content = content.replace('import logging', 'import logging\nfrom agent.utils.logger import get_logger')
            
        content = content.replace('logging.basicConfig(level=logging.INFO)', '')
        content = content.replace('logging.basicConfig(level=logging.DEBUG)', '')
        
        if content != original:
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"Updated {filepath}")
