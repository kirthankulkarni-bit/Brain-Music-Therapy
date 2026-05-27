import threading
import time

# global variables that both threads can see
# next_audio_prompt holds the instructions for the background worker
# audio_buffer holds the completed audio file ready for playback
next_audio_prompt = "calm ambient music"
audio_buffer = "initial_silent_loop.wav"
system_running = True

def background_audio_generator():
    """this function runs entirely in the background to handle heavy lifting"""
    global audio_buffer, next_audio_prompt, system_running
    
    while system_running:
        # save the current prompt so it doesn't change mid-generation
        current_job_prompt = next_audio_prompt
        print(f"\n[BACKGROUND] starting generation for: '{current_job_prompt}'...")
        
        # simulate the 3 seconds it takes musicgen to generate audio
        time.sleep(3) 
        
        # update the buffer with the fresh fake file name
        audio_buffer = f"completed_{current_job_prompt.replace(' ', '_')}.wav"
        print(f"[BACKGROUND] finished! buffer updated to: {audio_buffer}")
        
        # sleep briefly to avoid burning cpu cycles when no new prompt is needed
        time.sleep(1)

# step 1: spin up the background thread and start it
bg_thread = threading.Thread(target=background_audio_generator, daemon=True)
bg_thread.start()

# step 2: run the main playback loop
print("--- starting main audio playback loop ---")
try:
    for loop_count in range(1, 4):
        print(f"\n[MAIN] playing track: {audio_buffer} (duration: 5 seconds)")
        
        # simulate playing a 5-second loop
        # during these 5 seconds, the background thread is generating the next file
        time.sleep(5) 
        
        # change the prompt halfway through to simulate a shifting brain state
        if loop_count == 1:
            next_audio_prompt = "high energy synthwave"
            print("\n[MAIN] brain state changed! updated next prompt instruction")

except KeyboardInterrupt:
    print("\nstopping system...")

system_running = False
print("--- system stopped ---")