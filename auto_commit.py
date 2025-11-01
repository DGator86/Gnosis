#!/usr/bin/env python3
"""
Gnosis Auto-Commit Wrapper
Quick commands for push-as-you-go development
"""

import sys
import os
from git_autopush import GitAutoPush

def commit_agent3():
    """Commit Agent 3 completion"""
    git = GitAutoPush()
    
    files = [
        'agent3_sentiment.py',
        'test_agent3_comprehensive.py'
    ]
    
    description = "Seven regime classification, behavioral bias detection, hysteresis, sentiment divergence analysis"
    
    success = git.create_component_commit("Agent3-Sentiment", files, description)
    
    if success:
        print("🎉 Agent 3 committed and pushed!")
        git.status()
    else:
        print("❌ Agent 3 commit failed")

def commit_dhpe():
    """Commit DHPE Engine"""
    git = GitAutoPush()
    
    files = [
        'dhpe_engine.py'
    ]
    
    description = "Options market microstructure analysis with gamma exposure, hedge pressure, max pain"
    
    success = git.create_component_commit("DHPE-Engine", files, description)
    
    if success:
        print("🎉 DHPE Engine committed and pushed!")
    else:
        print("❌ DHPE commit failed")

def commit_agent2():
    """Commit Agent 2 Liquidity Analyzer"""
    git = GitAutoPush()
    
    files = [
        'agent2_advanced_liquidity.py'
    ]
    
    description = "Advanced liquidity analysis with volume profiles, VWAP, support/resistance detection"
    
    success = git.create_component_commit("Agent2-Liquidity", files, description)
    
    if success:
        print("🎉 Agent 2 committed and pushed!")
    else:
        print("❌ Agent 2 commit failed")

def commit_autopush_system():
    """Commit the auto-push system itself"""
    git = GitAutoPush()
    
    files = [
        'git_autopush.py',
        'auto_commit.py',
        '.gitignore'
    ]
    
    description = "Automated git workflow for push-as-you-go development"
    
    success = git.create_component_commit("Git-AutoPush-System", files, description)
    
    if success:
        print("🎉 Auto-push system committed!")
    else:
        print("❌ Auto-push system commit failed")

def setup_remote(remote_url: str):
    """Setup remote repository URL"""
    git = GitAutoPush()
    
    if git.setup_repo(remote_url):
        print(f"✅ Remote setup complete: {remote_url}")
        
        # Commit current state
        files = ['git_autopush.py', 'auto_commit.py']
        git.create_gitignore()
        files.append('.gitignore')
        
        success = git.create_component_commit("Initial-Setup", files, 
                                            "Git auto-push system and initial configuration")
        if success:
            print("🎉 Initial setup committed and pushed!")
    else:
        print("❌ Remote setup failed")

def commit_fix(component: str, files: list, description: str):
    """Generic fix commit"""
    git = GitAutoPush()
    
    success = git.create_fix_commit(component, files, description)
    
    if success:
        print(f"🎉 {component} fix committed and pushed!")
        git.status()
    else:
        print(f"❌ {component} fix commit failed")

def quick_status():
    """Show git status"""
    git = GitAutoPush()
    git.status()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("""
Gnosis Auto-Commit Commands:

  agent3        - Commit Agent 3 Sentiment Interpreter  
  agent2        - Commit Agent 2 Liquidity Analyzer
  dhpe          - Commit DHPE Engine
  autopush      - Commit the auto-push system
  status        - Show git status
  setup <url>   - Setup remote repository
  
Examples:
  python auto_commit.py agent3
  python auto_commit.py setup https://github.com/user/gnosis.git
  python auto_commit.py status
        """)
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "agent3":
        commit_agent3()
    elif command == "agent2":
        commit_agent2()
    elif command == "dhpe":
        commit_dhpe()
    elif command == "autopush":
        commit_autopush_system()
    elif command == "status":
        quick_status()
    elif command == "setup" and len(sys.argv) > 2:
        setup_remote(sys.argv[2])
    else:
        print(f"❌ Unknown command: {command}")
        sys.exit(1)