"""
🚀 Automated Colab Training Script
This enhanced training script supports webhook integration for pipeline automation.

Features:
- Webhook endpoint for remote training triggers
- Progress reporting via GitHub API
- Automatic result upload
- Status notifications

Usage in Colab:
    # Method 1: Direct execution
    exec(open('colab_automated.py').read())
    
    # Method 2: With webhook server
    !python colab_automated.py --enable_webhook --port 8080
"""

import json
import os
import time
import requests
from datetime import datetime
import threading
from flask import Flask, request, jsonify
import argparse

class ColabAutomatedTraining:
    def __init__(self, github_token=None, repo_owner="prashant-2050", repo_name="Ai-practise"):
        self.github_token = github_token or os.environ.get('GITHUB_TOKEN')
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.training_status = "idle"
        self.training_progress = {}
        
    def setup_environment(self):
        """Set up Colab environment for training"""
        print("🔧 Setting up Colab environment...")
        
        # Clone repository if not exists
        if not os.path.exists('Ai-practise'):
            os.system(f"git clone https://github.com/{self.repo_owner}/{self.repo_name}.git")
            os.chdir('Ai-practise')
        else:
            os.chdir('Ai-practise')
            os.system("git pull origin main")
        
        # Install dependencies
        os.system("pip install -q torch torchvision torchaudio")
        os.system("pip install -q -r requirements.txt")
        os.system("pip install -q flask ngrok")
        
        # Download data
        if not os.path.exists('shakespeare.txt'):
            os.system("wget -q -O shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
        
        # Create directories
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        print("✅ Environment ready!")
        
    def start_training(self, config):
        """Start training with given configuration"""
        self.training_status = "running"
        self.training_progress = {
            'start_time': datetime.now().isoformat(),
            'config': config,
            'current_step': 0,
            'current_loss': 0,
            'best_loss': float('inf')
        }
        
        print(f"🚀 Starting training: {config}")
        
        try:
            # Build training command
            cmd_parts = [
                "python", "train_cloud.py",
                "--model_size", config.get('model_size', 'micro'),
                "--max_steps", str(config.get('max_steps', 3000)),
                "--batch_size", str(config.get('batch_size', 64)),
                "--device", config.get('device', 'cuda'),
                "--learning_rate", str(config.get('learning_rate', 3e-4)),
                "--eval_interval", str(config.get('eval_interval', 200)),
                "--save_interval", str(config.get('save_interval', 500))
            ]
            
            if config.get('mixed_precision', True):
                cmd_parts.extend(["--mixed_precision", "true"])
            
            # Execute training
            import subprocess
            process = subprocess.Popen(
                cmd_parts,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Monitor training output
            for line in process.stdout:
                print(line.strip())
                self.parse_training_output(line)
                
                # Update progress periodically
                if self.training_progress['current_step'] % 50 == 0:
                    self.report_progress()
            
            process.wait()
            
            if process.returncode == 0:
                self.training_status = "completed"
                print("✅ Training completed successfully!")
                
                # Auto-upload if requested
                if config.get('auto_upload', True):
                    self.upload_results()
                
                return True
            else:
                self.training_status = "failed"
                print("❌ Training failed!")
                return False
                
        except Exception as e:
            self.training_status = "failed"
            print(f"❌ Training error: {e}")
            return False
    
    def parse_training_output(self, line):
        """Parse training output to extract progress"""
        try:
            if "Step" in line and "Loss:" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "Step":
                        self.training_progress['current_step'] = int(parts[i+1])
                    elif part == "Loss:":
                        self.training_progress['current_loss'] = float(parts[i+1])
                        
            elif "Eval Loss:" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "Loss:":
                        eval_loss = float(parts[i+1])
                        if eval_loss < self.training_progress['best_loss']:
                            self.training_progress['best_loss'] = eval_loss
                            
        except Exception as e:
            # Ignore parsing errors
            pass
    
    def report_progress(self):
        """Report training progress to GitHub or webhook"""
        try:
            progress_data = {
                'status': self.training_status,
                'progress': self.training_progress,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save progress locally
            with open('logs/training_progress.json', 'w') as f:
                json.dump(progress_data, f, indent=2)
            
            # TODO: Send to GitHub API or webhook
            # if self.github_token:
            #     self.update_github_status(progress_data)
                
        except Exception as e:
            print(f"⚠️ Error reporting progress: {e}")
    
    def upload_results(self):
        """Upload training results to GitHub"""
        print("📤 Uploading results to GitHub...")
        
        try:
            # Configure git
            os.system("git config --global user.name 'Colab Training Bot'")
            os.system("git config --global user.email 'colab@training.bot'")
            
            # Add results
            os.system("git add checkpoints/ logs/")
            
            # Create commit message
            config = self.training_progress.get('config', {})
            model_size = config.get('model_size', 'unknown')
            max_steps = config.get('max_steps', 'unknown')
            best_loss = self.training_progress.get('best_loss', 'unknown')
            
            commit_msg = f"🎯 Colab trained model - {model_size} model, {max_steps} steps, loss {best_loss:.4f}"
            
            # Commit and push
            os.system(f'git commit -m "{commit_msg}"')
            
            # Push to main branch
            push_result = os.system("git push origin main")
            
            if push_result == 0:
                print("✅ Results uploaded successfully!")
                return True
            else:
                print("❌ Failed to push to GitHub")
                return False
                
        except Exception as e:
            print(f"❌ Upload error: {e}")
            return False
    
    def create_webhook_server(self, port=8080):
        """Create Flask webhook server for remote training triggers"""
        app = Flask(__name__)
        
        @app.route('/train', methods=['POST'])
        def webhook_train():
            try:
                config = request.json
                print(f"📨 Received training request: {config}")
                
                # Validate config
                if not config or 'model_size' not in config:
                    return jsonify({'error': 'Invalid config'}), 400
                
                # Start training in background thread
                training_id = f"train_{int(time.time())}"
                thread = threading.Thread(
                    target=self.start_training,
                    args=(config,)
                )
                thread.daemon = True
                thread.start()
                
                return jsonify({
                    'status': 'started',
                    'training_id': training_id,
                    'message': 'Training started successfully'
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @app.route('/status', methods=['GET'])
        def webhook_status():
            return jsonify({
                'status': self.training_status,
                'progress': self.training_progress
            })
        
        @app.route('/health', methods=['GET'])
        def webhook_health():
            return jsonify({'status': 'healthy', 'service': 'colab-training'})
        
        print(f"🌐 Starting webhook server on port {port}")
        print(f"📡 Endpoints:")
        print(f"   POST /train - Start training")
        print(f"   GET /status - Check status")
        print(f"   GET /health - Health check")
        
        # Expose via ngrok for external access
        try:
            from pyngrok import ngrok
            public_url = ngrok.connect(port)
            print(f"🔗 Public URL: {public_url}")
            print(f"📤 Use this URL for pipeline automation!")
        except Exception as e:
            print(f"⚠️ Could not create public tunnel: {e}")
            print(f"🏠 Local access only: http://localhost:{port}")
        
        app.run(host='0.0.0.0', port=port, debug=False)

def main():
    parser = argparse.ArgumentParser(description='Automated Colab Training')
    parser.add_argument('--enable_webhook', action='store_true',
                       help='Enable webhook server for remote triggers')
    parser.add_argument('--port', type=int, default=8080,
                       help='Webhook server port')
    parser.add_argument('--model_size', choices=['nano', 'micro', 'small'], default='micro',
                       help='Model size to train')
    parser.add_argument('--max_steps', type=int, default=3000,
                       help='Maximum training steps')
    parser.add_argument('--auto_upload', action='store_true', default=True,
                       help='Auto-upload results to GitHub')
    
    args = parser.parse_args()
    
    # Initialize training system
    trainer = ColabAutomatedTraining()
    trainer.setup_environment()
    
    if args.enable_webhook:
        # Start webhook server
        trainer.create_webhook_server(args.port)
    else:
        # Direct training
        config = {
            'model_size': args.model_size,
            'max_steps': args.max_steps,
            'batch_size': 64,
            'device': 'cuda',
            'mixed_precision': True,
            'auto_upload': args.auto_upload
        }
        
        success = trainer.start_training(config)
        exit(0 if success else 1)

if __name__ == "__main__":
    main()

# Quick execution block for Colab
if 'google.colab' in str(get_ipython()):
    print("🚀 Running in Google Colab - starting automated training...")
    
    # Default Colab configuration
    default_config = {
        'model_size': 'micro',
        'max_steps': 2000,
        'batch_size': 64,
        'device': 'cuda',
        'mixed_precision': True,
        'auto_upload': True
    }
    
    trainer = ColabAutomatedTraining()
    trainer.setup_environment()
    trainer.start_training(default_config)