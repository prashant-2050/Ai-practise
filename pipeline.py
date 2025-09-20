"""
🚀 Automated Training Pipeline
This script enables fully automated training and deployment pipeline:

Pipeline Flow:
1. Trigger training in Colab programmatically
2. Monitor training progress
3. Auto-upload results to GitHub
4. Trigger deployment workflow
5. Deploy to Hugging Face Spaces
6. Send notifications

Usage:
    python pipeline.py --model_size micro --deploy_after_training true
"""

import requests
import time
import json
import os
import subprocess
from datetime import datetime
import argparse

class AutomatedPipeline:
    def __init__(self, github_token=None, hf_token=None, webhook_url=None):
        self.github_token = github_token or os.environ.get('GITHUB_TOKEN')
        self.hf_token = hf_token or os.environ.get('HF_TOKEN')
        self.webhook_url = webhook_url or os.environ.get('WEBHOOK_URL')
        self.repo_owner = "prashant-2050"
        self.repo_name = "Ai-practise"
        
    def trigger_colab_training(self, model_size='micro', max_steps=3000):
        """
        Trigger Colab training via webhook or API
        """
        print(f"🚀 Triggering Colab training: {model_size} model, {max_steps} steps")
        
        # Colab webhook payload
        payload = {
            'model_size': model_size,
            'max_steps': max_steps,
            'batch_size': 64,
            'device': 'cuda',
            'mixed_precision': True,
            'auto_upload': True,
            'github_repo': f"{self.repo_owner}/{self.repo_name}",
            'branch': 'main'
        }
        
        if self.webhook_url:
            try:
                response = requests.post(self.webhook_url, json=payload, timeout=30)
                if response.status_code == 200:
                    training_id = response.json().get('training_id')
                    print(f"✅ Training started! ID: {training_id}")
                    return training_id
                else:
                    print(f"❌ Failed to trigger training: {response.status_code}")
                    return None
            except Exception as e:
                print(f"❌ Error triggering training: {e}")
                return None
        else:
            print("⚠️ No webhook URL configured. Please train manually in Colab.")
            print(f"📖 Instructions: Open https://colab.research.google.com/github/{self.repo_owner}/{self.repo_name}/blob/main/colab_training.ipynb")
            return "manual"
    
    def monitor_training(self, training_id, check_interval=60):
        """
        Monitor training progress via GitHub or webhook
        """
        if training_id == "manual":
            print("📊 Manual training mode - waiting for results...")
            return self.wait_for_manual_upload()
        
        print(f"📊 Monitoring training {training_id}...")
        start_time = time.time()
        
        while True:
            try:
                # Check for new commits or artifacts
                status = self.check_training_status(training_id)
                
                if status == 'completed':
                    print("✅ Training completed successfully!")
                    return True
                elif status == 'failed':
                    print("❌ Training failed!")
                    return False
                elif status == 'running':
                    elapsed = (time.time() - start_time) / 60
                    print(f"⏳ Training in progress... ({elapsed:.1f} minutes)")
                
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                print("⏹️ Monitoring stopped by user")
                return False
            except Exception as e:
                print(f"⚠️ Error monitoring: {e}")
                time.sleep(check_interval)
    
    def check_training_status(self, training_id):
        """
        Check training status via GitHub API
        """
        try:
            # Check recent commits for training results
            url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/commits"
            headers = {'Authorization': f'token {self.github_token}'} if self.github_token else {}
            
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                commits = response.json()
                for commit in commits[:5]:  # Check last 5 commits
                    message = commit['commit']['message'].lower()
                    if 'trained model' in message or 'colab training' in message:
                        # Check if this commit has model files
                        if self.check_commit_has_models(commit['sha']):
                            return 'completed'
                
            return 'running'
            
        except Exception as e:
            print(f"Error checking status: {e}")
            return 'running'
    
    def wait_for_manual_upload(self):
        """
        Wait for manual training completion and upload
        """
        print("🔄 Waiting for training completion and manual upload...")
        print("📋 Please:")
        print("   1. Complete training in Colab")
        print("   2. Run the upload cell to push results to GitHub")
        print("   3. Press Enter here when done")
        
        input("Press Enter when training is complete and uploaded...")
        return True
    
    def check_commit_has_models(self, sha):
        """
        Check if a commit contains model files
        """
        try:
            url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/git/trees/{sha}?recursive=1"
            headers = {'Authorization': f'token {self.github_token}'} if self.github_token else {}
            
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                tree = response.json()
                for item in tree.get('tree', []):
                    if item['path'].startswith('checkpoints/') and item['path'].endswith('.pt'):
                        return True
            return False
        except:
            return False
    
    def trigger_deployment(self, model_path='checkpoints/best_model.pt', model_name=None):
        """
        Trigger GitHub Actions deployment workflow
        """
        if not model_name:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f"llm-{timestamp}"
        
        print(f"🚀 Triggering deployment: {model_name}")
        
        if not self.github_token:
            print("⚠️ No GitHub token - manual deployment required")
            print(f"📖 Go to: https://github.com/{self.repo_owner}/{self.repo_name}/actions")
            return False
        
        # Trigger workflow via GitHub API
        url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/actions/workflows/deploy-hf-spaces.yml/dispatches"
        headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        payload = {
            'ref': 'main',
            'inputs': {
                'model_path': model_path,
                'model_name': model_name,
                'deploy_demo': 'true'
            }
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 204:
                print(f"✅ Deployment triggered for {model_name}")
                return True
            else:
                print(f"❌ Failed to trigger deployment: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error triggering deployment: {e}")
            return False
    
    def monitor_deployment(self, timeout_minutes=15):
        """
        Monitor deployment workflow progress
        """
        print("📊 Monitoring deployment...")
        start_time = time.time()
        
        while (time.time() - start_time) < (timeout_minutes * 60):
            try:
                # Check workflow runs
                url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/actions/runs"
                headers = {'Authorization': f'token {self.github_token}'} if self.github_token else {}
                
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    runs = response.json().get('workflow_runs', [])
                    
                    # Find recent deployment runs
                    for run in runs[:3]:
                        if 'deploy' in run['name'].lower():
                            status = run['status']
                            conclusion = run['conclusion']
                            
                            if status == 'completed':
                                if conclusion == 'success':
                                    print("✅ Deployment completed successfully!")
                                    return True
                                else:
                                    print(f"❌ Deployment failed: {conclusion}")
                                    return False
                            elif status == 'in_progress':
                                elapsed = (time.time() - start_time) / 60
                                print(f"⏳ Deployment in progress... ({elapsed:.1f} minutes)")
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                print(f"⚠️ Error monitoring deployment: {e}")
                time.sleep(30)
        
        print("⏰ Deployment monitoring timed out")
        return False
    
    def send_notification(self, status, model_name, details=None):
        """
        Send notification about pipeline completion
        """
        emoji = "✅" if status == "success" else "❌"
        message = f"{emoji} Pipeline {status}: {model_name}"
        
        if details:
            message += f"\n{details}"
        
        print(f"📢 {message}")
        
        # TODO: Add Slack/Discord webhook integration
        # if self.slack_webhook:
        #     requests.post(self.slack_webhook, json={'text': message})
    
    def run_full_pipeline(self, model_size='micro', max_steps=3000, auto_deploy=True):
        """
        Run the complete automated pipeline
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f"llm-{model_size}-{timestamp}"
        
        print("🚀 Starting Automated LLM Pipeline")
        print(f"📊 Model: {model_size}, Steps: {max_steps}")
        print(f"🏷️ Name: {model_name}")
        print("=" * 50)
        
        # Step 1: Start training
        training_id = self.trigger_colab_training(model_size, max_steps)
        if not training_id:
            print("❌ Failed to start training")
            return False
        
        # Step 2: Monitor training
        training_success = self.monitor_training(training_id)
        if not training_success:
            print("❌ Training failed or was cancelled")
            self.send_notification("failed", model_name, "Training failed")
            return False
        
        # Step 3: Deploy if requested
        if auto_deploy:
            deployment_success = self.trigger_deployment(model_name=model_name)
            if deployment_success:
                deployment_complete = self.monitor_deployment()
                if deployment_complete:
                    hf_url = f"https://huggingface.co/{self.repo_owner}/{model_name}"
                    self.send_notification("success", model_name, f"Available at: {hf_url}")
                    print(f"🎉 Pipeline complete! Model available at: {hf_url}")
                    return True
                else:
                    self.send_notification("partial", model_name, "Training succeeded, deployment failed")
                    return False
            else:
                print("⚠️ Training succeeded but deployment failed")
                return False
        else:
            self.send_notification("success", model_name, "Training completed (no deployment)")
            print("✅ Training pipeline complete!")
            return True

def main():
    parser = argparse.ArgumentParser(description='Automated LLM Training Pipeline')
    parser.add_argument('--model_size', choices=['nano', 'micro', 'small'], default='micro',
                       help='Model size to train')
    parser.add_argument('--max_steps', type=int, default=3000,
                       help='Maximum training steps')
    parser.add_argument('--auto_deploy', type=bool, default=True,
                       help='Automatically deploy after training')
    parser.add_argument('--github_token', type=str,
                       help='GitHub token for API access')
    parser.add_argument('--webhook_url', type=str,
                       help='Webhook URL for Colab integration')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = AutomatedPipeline(
        github_token=args.github_token,
        webhook_url=args.webhook_url
    )
    
    # Run the pipeline
    success = pipeline.run_full_pipeline(
        model_size=args.model_size,
        max_steps=args.max_steps,
        auto_deploy=args.auto_deploy
    )
    
    if success:
        print("🎉 Automated pipeline completed successfully!")
        exit(0)
    else:
        print("❌ Pipeline failed")
        exit(1)

if __name__ == "__main__":
    main()