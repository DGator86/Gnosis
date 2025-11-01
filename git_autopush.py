#!/usr/bin/env python3
"""
Gnosis Git Auto-Push System
Automatically commits and pushes changes after each completed component
"""

import subprocess
import os
import sys
from datetime import datetime
from typing import List, Optional

class GitAutoPush:
    """Automated Git workflow for push-as-you-go development"""
    
    def __init__(self, repo_path: str = "/home/user"):
        self.repo_path = repo_path
        self.remote_url = None
        self.branch = "main"
        
    def setup_repo(self, remote_url: Optional[str] = None) -> bool:
        """Initialize or verify git repository setup"""
        
        os.chdir(self.repo_path)
        
        try:
            # Check if git repo exists
            result = subprocess.run(['git', 'status'], capture_output=True, text=True)
            if result.returncode != 0:
                print("❌ Not in a git repository")
                return False
            
            # Set up remote if provided
            if remote_url:
                self.remote_url = remote_url
                try:
                    # Add or update remote
                    subprocess.run(['git', 'remote', 'remove', 'origin'], capture_output=True)
                    subprocess.run(['git', 'remote', 'add', 'origin', remote_url], 
                                 capture_output=True, check=True)
                    print(f"✅ Remote origin set to: {remote_url}")
                except subprocess.CalledProcessError:
                    print(f"⚠️  Could not set remote: {remote_url}")
            
            # Switch to main branch
            try:
                subprocess.run(['git', 'branch', '-M', 'main'], capture_output=True, check=True)
                self.branch = "main"
                print("✅ Using main branch")
            except subprocess.CalledProcessError:
                print("⚠️  Could not switch to main branch")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Git setup failed: {e}")
            return False
    
    def create_gitignore(self) -> None:
        """Create comprehensive .gitignore for Python/ML project"""
        
        gitignore_content = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyInstaller
*.manifest
*.spec

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Project specific
*.log
*.csv
*.json
config/secrets/
data/raw/
data/processed/
models/
checkpoints/
logs/
temp/
*.tmp

# Jupyter Notebook
.ipynb_checkpoints

# Financial data
*.pkl
market_data/
options_data/
backtests/results/

# API keys and credentials
api_keys.py
credentials.json
.env.local
.env.production
"""
        
        gitignore_path = os.path.join(self.repo_path, '.gitignore')
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content)
        
        print("✅ Created .gitignore")
    
    def commit_and_push(self, 
                       files: List[str], 
                       message: str, 
                       component_name: str,
                       push_to_remote: bool = True) -> bool:
        """
        Commit specific files and push to remote
        
        Args:
            files: List of file paths to commit
            message: Commit message
            component_name: Name of completed component
            push_to_remote: Whether to push to remote (requires remote setup)
        """
        
        os.chdir(self.repo_path)
        
        try:
            # Add specific files
            for file_path in files:
                if os.path.exists(file_path):
                    subprocess.run(['git', 'add', file_path], check=True)
                    print(f"✅ Added: {file_path}")
                else:
                    print(f"⚠️  File not found: {file_path}")
            
            # Check if there are changes to commit
            result = subprocess.run(['git', 'diff', '--staged'], 
                                  capture_output=True, text=True)
            if not result.stdout.strip():
                print("ℹ️  No changes to commit")
                return True
            
            # Create commit with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            full_message = f"[{component_name}] {message}\n\nAuto-committed: {timestamp}"
            
            subprocess.run(['git', 'commit', '-m', full_message], check=True)
            print(f"✅ Committed: {message}")
            
            # Push to remote if configured
            if push_to_remote and self.remote_url:
                try:
                    # Try to push
                    result = subprocess.run(['git', 'push', 'origin', self.branch], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        print(f"✅ Pushed to remote: {self.branch}")
                    else:
                        # If push fails, try force push (for initial setup)
                        subprocess.run(['git', 'push', '-u', 'origin', self.branch], 
                                     capture_output=True, check=True)
                        print(f"✅ Force pushed to remote: {self.branch}")
                        
                except subprocess.CalledProcessError as e:
                    print(f"⚠️  Push failed: {e}")
                    print("Commit successful but not pushed to remote")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Git operation failed: {e}")
            return False
    
    def create_component_commit(self, component_name: str, files: List[str], 
                              description: str = "") -> bool:
        """
        Convenience method for committing completed components
        
        Args:
            component_name: Name of the component (e.g., "Agent3", "DHPE-Engine")
            files: List of files to include
            description: Optional description of what was implemented
        """
        
        if description:
            message = f"Implement {component_name}: {description}"
        else:
            message = f"Complete {component_name} implementation"
        
        return self.commit_and_push(files, message, component_name)
    
    def create_fix_commit(self, component_name: str, files: List[str], 
                         fix_description: str) -> bool:
        """
        Convenience method for committing fixes
        """
        
        message = f"Fix {component_name}: {fix_description}"
        return self.commit_and_push(files, message, f"{component_name}-Fix")
    
    def status(self) -> None:
        """Show git repository status"""
        
        os.chdir(self.repo_path)
        
        try:
            # Show status
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True)
            
            if result.stdout.strip():
                print("📋 Uncommitted changes:")
                for line in result.stdout.strip().split('\n'):
                    print(f"   {line}")
            else:
                print("✅ Working directory clean")
            
            # Show recent commits
            result = subprocess.run(['git', 'log', '--oneline', '-5'], 
                                  capture_output=True, text=True)
            if result.stdout.strip():
                print("\n📜 Recent commits:")
                for line in result.stdout.strip().split('\n'):
                    print(f"   {line}")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Could not get status: {e}")

def demo_autopush():
    """Demo the auto-push system"""
    
    print("=== Gnosis Git Auto-Push Demo ===")
    
    git = GitAutoPush()
    
    # Setup repository
    if git.setup_repo():
        print("✅ Repository ready")
    else:
        print("❌ Repository setup failed")
        return
    
    # Create .gitignore
    git.create_gitignore()
    
    # Show status
    print("\n📋 Current Status:")
    git.status()
    
    return git

if __name__ == "__main__":
    demo_autopush()