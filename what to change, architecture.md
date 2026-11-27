I want to significantly improve the automatic clip boundary selection in my existing OpenShorts project.

CONTEXT:
I have integrated or am integrating local WhisperX at ~/Projects/personal/whisperx for transcription and word-level timestamp alignment. OpenShorts currently identifies interesting moments using an LLM, but its generated clips frequently have poor start and end timestamps. I have to manually adjust almost every clip.

GOAL:
Build a Smart Clip Boundary Refinement system that automatically produces complete, natural, publishable clips with minimal manual editing.

IMPORTANT: First inspect the existing OpenShorts architecture, highlight detection logic, transcript data structures, timestamp formats, FFmpeg rendering pipeline and frontend clip editor. Understand how everything currently works before modifying anything. Do not rewrite the application.

IMPLEMENT THE FOLLOWING:

1. TRANSCRIPT-AWARE BOUNDARY REFINEMENT
- Use WhisperX word-level timestamps and transcript segments.
- Identify sentence boundaries using punctuation and transcript timing.
- Snap proposed clip starts to sensible sentence or phrase boundaries.
- Ensure the clip does not start halfway through a spoken sentence.
- Ensure the final sentence and interesting moment are completed.
- Preserve the original video timestamp offsets accurately.

2. CONTEXTUAL PADDING
- Add configurable start and end padding.
- Default to approximately 2 seconds before and after the relevant speech.
- Avoid unnecessary silence while preserving natural pauses.
- Never remove speech simply to meet a duration target.

3. AUDIO-AWARE REFINEMENT
- Use FFmpeg audio analysis or silence detection to identify natural pauses.
- Detect speech cutoffs and abrupt endings where possible.
- Do not blindly remove all pauses.
- Avoid introducing awkward audio transitions.

4. LLM-BASED CLIP QUALITY EVALUATION
- Evaluate the proposed clip using the transcript surrounding the selected timestamps.
- Check whether the clip has a meaningful beginning, sufficient context, a complete interesting moment and a natural ending.
- If the proposed boundaries are poor, refine them using available transcript timestamps.
- Do not invent transcript content or timestamps.
- Avoid unnecessary LLM calls; prefer deterministic boundary refinement where possible.
- Keep the existing LLM provider configurable.

5. DURATION MANAGEMENT
- Configurable minimum duration: 20 seconds.
- Preferred duration: 30–60 seconds.
- Configurable maximum duration: 60 seconds.
- Do not force clips to reach a specific duration.
- Reject or flag clips that cannot be made coherent within the configured limits.

6. CLIP QUALITY METADATA
- Store original AI-selected timestamps and refined timestamps separately.
- Record why boundaries were changed.
- Add quality warnings for missing context, incomplete sentences, speech cutoffs or excessive silence.
- Make the refinement process reproducible.

7. FRONTEND
- Preserve the existing clip editor and user workflow.
- Display the original and refined timestamps where appropriate.
- Allow users to disable automatic refinement.
- Allow manual overrides.
- Do not remove existing editing capabilities.

8. TESTING
- Create tests for clips beginning mid-sentence, clips ending mid-sentence, incomplete punchlines, natural pauses, long silence, short clips and timestamp offsets.
- Test that subtitles remain synchronized after trimming.
- Use sample files and mock LLM responses where appropriate.

IMPORTANT CONSTRAINTS:
- Do not change the existing highlight discovery strategy unnecessarily.
- Do not break WhisperX integration.
- Do not replace existing video rendering functionality without justification.
- Do not introduce paid APIs or external services.
- Keep the project local.
- Do not deploy.
- Do not make unrelated UI or architectural changes.

FIRST: Report the existing clipping architecture and your proposed implementation plan. Wait for my approval before making code changes.

After approval, implement this incrementally and report changed files, tests and remaining limitations.