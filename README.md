# MoveMatch
## Purpose
Targeted at recreational dancers who need more individualized and structured practices in preparing for a performance.

## Personal motivation 
Throughout my performance preparation journey at NUS High, cleaning movements and synchronizing dancers have always been the most time-consuming (more than 100 hours for a 4.5-minute performance!!) due to the mismatch between how the instructors' (my co-leaders and I) view certain moves and give corresponding instructions and the dancers' anatomy or even their differing interpretations of certain instructions.

MoveMatch intends to give objective numerical evaluations that standardize each dancer to the same reference dance video, allowing each individual to have their tailored practice. Ultimately, the project hopes to reduce the instructors' workload to bringing the group performance together, increasing the effectiveness group practices.

## Demo
Pending!

## Screenshots
![Alt Text](assets/screenshot_1.png)
![Alt Text](assets/screenshot_2.png)
![Alt Text](assets/screenshot_3.png)
![Alt Text](assets/screenshot_4.png)
![Alt Text](assets/screenshot_5.png)
![Alt Text](assets/screenshot_6.png)

## Contributions
### Pose-tracking


## Technical decisions
* Due to limited computational resources, the functionality is limited to single-dancer use

Model options:
1. Google AI's pose landmarker lite (current)
2. HybrIK: Hybrid analytical-neural inverse kinematics for body mesh recovery (cannot install all required setup on Mac)

Project flow:
1. Homepage --> User have options: Upload new dance video | Access past dance video
2.1 Upload new dance video
--> Have an option to navigate to practice page, which offers: Frame by frame playing, slowed down video run
--> After confident with frame-by-frame practice, attempt speed-variant ghost mode practice (0.25x, 0.5x, 0.75x)
--> After ghost mode practice, attempt evaluation (scoring)
For now, assume that the app is single-user