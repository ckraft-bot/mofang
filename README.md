# 魔方
Intent: Recently my brother and I talked about computer vision which inspired me to try more computer vision projects.  Previously, I created motion and face detection for a bigger home security system project.  My high school math mentioned I should program some sort of simulator to solve the Rubik's cube. I can solve the cube much faster than I can program a machine to solve the cube.  Now that my coding is a little more advanced, I'd like to take on this challenge. I think OpenCV should be sufficient for this task.

OpenCV (Open Source Computer Vision Library) is an open source computer vision and machine learning software library. 
    OpenCV was built to provide a common infrastructure for computer vision applications and to accelerate the use of machine perception in the commercial products. 
    Being a BSD-licensed product, OpenCV makes it easy for businesses to utilize and modify the code.
    The library has more than 2500 optimized algorithms, which includes a comprehensive set of both classic and state-of-the-art computer vision and machine learning algorithms. 
    These algorithms can be used to detect and recognize faces, identify objects, classify human actions in videos, track camera movements, track moving objects, 
    extract 3D models of objects, produce 3D point clouds from stereo cameras, stitch images together to produce a high resolution image of an entire scene, 
    find similar images from an image database, remove red eyes from images taken using flash, follow eye movements, recognize scenery,
    and establish markers to overlay it with augmented reality, etc. 
    OpenCV has more than 47 thousand people of user community and estimated number of downloads exceeding 18 million. 
    The library is used extensively in companies, research groups and by governmental bodies.
    (Source: https://opencv.org/about/)

According to the WCA (World Cubing Association: https://www.worldcubeassociation.org/) scrambling is done with white on top and green in front (article 4d1).
The [Classic] Rubik’s Cube [3x3] was invented in 1974 by Ernõ Rubik, a Hungarian architecture professor. 
Rubik created the Cube as a learning exercise to teach his students about 3-dimensional spaces. 
Little did he know his “Magic Cube” (as he originally named it) would become one of world’s most famous puzzles of all time!
By the 1980’s, the Rubik’s Cube was a worldwide craze selling millions of Cubes every year and cementing its legacy into pop culture. 
Featured in The Simpsons, The Big Bang Theory, a Spice Girls music video, and major Hollywood movies, the popularity of the Rubik’s Cube continued to grow around the world. 
(source: https://rubiks.com/)

The notations of the classic Rubik's Cube are:

        Slices:
            F = front           F' = front prime
            B = back            B' = back prime
            U = upper           U' = upper prime
            D = down            D' = down prime
            L = left            L' = left prime
            R = right           R' = right prime
            M = middle (turns like L)       M' = middle prime
            E = equator (turns like D)      E' = equator prime
            S = standing (turns like F)     S' = standing prime
            
        Whole cube:
            imagine your cube on a 3D plane
            X = the whole cube moves like R
            Y = the whole cube moves like U
            Z = the whole cube moves like F
        
Rules on turns:  
The apostrophe denotes the surfaces of the cube turns in counter clockwise direction, otherwise assume clockwise turns.
Also all turns are 90 degreee turns unless specified otherwise.
2 before or after a notation means two rotations (180 degreee turns)--R2 = two right turns, RW2 - two wide right turns
W after a notation means wide moves--RW = the two most outer right slices are turned 90 degrees together

About the Kociemba Alg/ Two-Phase Alg:
The process divides the whole entity into smaller groupings thus constraining possibilities.
Group 0 = all available possibilities, Group 1 divides the cube in half, Group 2 the other half hence the name two-phase method.


Read more here:

1. http://kociemba.org/math/twophase.htm
2. https://www.speedsolving.com/wiki/index.php/Kociemba%27s_Algorithm

Dependencies
- kociemba
- opencv-python

