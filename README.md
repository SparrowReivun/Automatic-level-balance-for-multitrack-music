# Automatic-level-balance-for-multitrack-music

Input  : AIFF, FLAC, MP3, OGG, and WAV files across all platforms (file/folder).

Output : The final mixdown of multi-track audio files and stereo through 3 different level balance processes.


Created by jinjie.shi@qmul.ac.uk on 2024/Feb/26
Version 5.2 (March 10-25, 2024): Enhanced robustness when dealing with complex audio. Introduced a safety gain feature to 
ensure that gain processing does not increase peak amplitude beyond -1dB. Added active audio detection to avoid issues 
with infinite gain when processing inputs containing silent audio segments.

Version 5.3 (March 26 to present):
After testing over ten multitrack projects with more than 200 individual tracks, the system has shown remarkable stability. 
We are now focusing on enhancing the artistic expression of the system. An experienced mix engineer has joined me in reviewing 
the system's performance. We are marking tracks with inappropriate level balances in red. Currently, we have noticed that 
vocals, bass, and electric guitar levels are slightly lower than desired. We have found that setting intervals based on 
spectral centroid and spectral bandwidth is more appropriate. I have already incorporated filtering and level balance 
processing based on spectral centroid and bandwidth, and we are currently testing with new material.
C
#sr is same, don't know when bit depth change from 24bit to 16
"""
