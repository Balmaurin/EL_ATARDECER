
import asyncio
import sys
import os
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

async def verify_tier_2():
    print("🚀 Verifying Tier 2 Fixes...")
    results = {}

    try:
        print("\n1. Verifying enhanced_ai_threat_detector.py...")
        from packages.sheily_core.src.sheily_core.security.advanced.ai_security.enhanced_ai_threat_detector import EnhancedAIThreatDetector, InputMethod
        detector = EnhancedAIThreatDetector()
        
        # Test simulated speech
        res_speech = detector.process_multimodal_input("mock_audio:Hello World", InputMethod.SPEECH)
        print(f"   ✅ Speech simulation result: {res_speech.threat_detected} (Transcription: {res_speech.detected_patterns})") # Note: transcription is not in detected_patterns, but logged
        
        # Test simulated vision
        # Create dummy image file
        with open("test_image.jpg", "wb") as f:
            f.write(b"dummy image content")
        
        with open("test_image.jpg", "rb") as f:
            img_data = f.read()
            res_vision = detector.process_multimodal_input(img_data, InputMethod.VISION)
            print(f"   ✅ Vision simulation result: {res_vision.threat_detected}")
            
        os.remove("test_image.jpg")
        results['enhanced_ai_threat_detector'] = 'PASS'
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        results['enhanced_ai_threat_detector'] = f'FAIL: {e}'

    try:
        print("\n2. Verifying backup_manager.py...")
        from packages.sheily_core.src.sheily_core.backup.backup_manager import BackupManager, BackupConfig
        
        # Setup test env
        test_dir = Path("test_backup_data")
        test_dir.mkdir(exist_ok=True)
        (test_dir / "file1.txt").write_text("content1")
        (test_dir / "subdir").mkdir(exist_ok=True)
        (test_dir / "subdir" / "file2.txt").write_text("content2")
        
        config = BackupConfig(backup_dir="test_backups")
        manager = BackupManager(config)
        
        # Override critical components for test
        manager.critical_components = {"test_comp": [str(test_dir)]}
        
        # Create backup
        print("   Creating backup...")
        backup_meta = await manager.create_backup("full", components=["test_comp"])
        print(f"   ✅ Backup created: {backup_meta.id}")
        
        # Modify files
        (test_dir / "file1.txt").write_text("modified")
        
        # Restore backup
        print("   Restoring backup...")
        await manager.restore_backup(backup_meta.id)
        
        # Verify content
        content1 = (test_dir / "file1.txt").read_text()
        print(f"   ✅ Restored content: {content1}")
        
        if content1 == "content1":
            results['backup_manager'] = 'PASS'
        else:
            results['backup_manager'] = 'FAIL: Content mismatch'
            
        # Cleanup
        shutil.rmtree(test_dir)
        shutil.rmtree("test_backups")
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        results['backup_manager'] = f'FAIL: {e}'

    try:
        print("\n3. Verifying metacognicion.py...")
        from packages.consciousness.src.conciencia.modulos.metacognicion import MetacognitionEngine
        engine = MetacognitionEngine()
        
        evidence = [
            {'source_reliability': 0.8, 'strength': 0.9},
            {'source_reliability': 0.6, 'strength': 0.5}
        ]
        score = engine.calculate_evidence_score({}, evidence)
        print(f"   ✅ calculate_evidence_score result: {score}")
        
        # Test integration in assess_reasoning_quality
        quality = engine._assess_reasoning_quality([{'data': 'test', 'evidence': {'source_reliability': 0.9, 'strength': 0.8}}])
        print(f"   ✅ _assess_reasoning_quality result: {quality}")
        
        results['metacognicion'] = 'PASS'
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        results['metacognicion'] = f'FAIL: {e}'

    print("\n📊 Verification Summary:")
    for file, status in results.items():
        print(f"  {file}: {status}")

if __name__ == "__main__":
    asyncio.run(verify_tier_2())
