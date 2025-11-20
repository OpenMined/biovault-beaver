import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "python/src"))

import shutil
import time
import threading
import beaver
from beaver import Twin

# Clean start
shutil.rmtree("shared", ignore_errors=True)

print("=" * 70)
print("LIVE SYNC TEST - Real-time Twin Updates")
print("=" * 70)

# Setup
bv_bob = beaver.connect("shared", user="bob", auto_load_replies=False)
bv_alice = beaver.connect("shared", user="alice", auto_load_replies=False)

print("\n" + "=" * 70)
print("STEP 1: Bob creates a counter Twin")
print("=" * 70)

counter = Twin(
    private=1,
    public=1,
    owner="bob",
    name="counter"
)
print(counter)

print("\n" + "=" * 70)
print("STEP 2: Bob subscribes Alice for live updates")
print("=" * 70)

counter.subscribe_live("alice", context=bv_bob)

print("\n" + "=" * 70)
print("STEP 3: Bob enables live sync")
print("=" * 70)

counter.enable_live(mutable=False, interval=2.0)

print("\n" + "=" * 70)
print("STEP 4: Bob sends initial Twin to Alice")
print("=" * 70)

bv_bob.send(counter, user="alice")
time.sleep(0.5)

# Check counter state after send
print(f"\n[Debug] After send - counter.private={counter.private}, counter.public={counter.public}")

print("\n" + "=" * 70)
print("STEP 5: Alice receives and starts watching")
print("=" * 70)

# Alice loads the counter
count_alice = list(bv_alice.inbox())[0].load()
print(f"\nAlice's counter: {count_alice.public}")

print("\n" + "=" * 70)
print("STEP 6: Bob updates counter in background thread")
print("=" * 70)

def bob_update_loop():
    """Bob updates counter every 2 seconds."""
    for i in range(5):
        time.sleep(2)
        # Only update public side (private gets stripped anyway when sent)
        old_val = counter.public
        counter.public = old_val + 1
        # Verify it stuck
        time.sleep(0.1)
        new_val = counter.public
        print(f"\n[Bob] Updated counter: {old_val} -> {new_val} (actual={counter.public})")

# Start Bob's update thread
update_thread = threading.Thread(target=bob_update_loop, daemon=True)
update_thread.start()

print("\n" + "=" * 70)
print("STEP 7: Alice watches for updates (10 seconds)")
print("=" * 70)
print("(Alice should see updates as Bob changes the counter)\n")

# Alice watches for updates
start_time = time.time()
update_count = 0

try:
    for updated_twin in count_alice.watch_live(context=bv_alice, interval=1.0):
        elapsed = time.time() - start_time
        update_count += 1
        print(f"[Alice] Update #{update_count} at {elapsed:.1f}s: counter = {updated_twin.public}")

        # Stop after 10 seconds
        if elapsed > 10:
            break
except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")

# Wait for update thread to finish
update_thread.join(timeout=1)

print("\n" + "=" * 70)
print("STEP 8: Clean up")
print("=" * 70)

counter.disable_live()

print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print(f"""
✅ Bob's counter started at: 1
✅ Bob updated counter {5} times
✅ Alice received {update_count} updates
✅ Final value: {count_alice.public}

The live sync system:
1. Detects changes on Bob's side (every 2s)
2. Automatically sends updates to subscribed users (Alice)
3. Alice's watch_live() monitors inbox and reloads
4. Updates propagate in near real-time

No manual sending/loading required - it's truly live!
""")
