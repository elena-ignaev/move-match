# MoveMatch
## Purpose
Targeted at recreational dancers who need more individualized and structured practices in preparing for a performance.

## Personal motivation 
Throughout my performance preparation journey at NUS High, cleaning movements and synchronizing dancers have always been the most time-consuming (more than 100 hours for a 4.5-minute performance!!) due to the mismatch between how the instructors' (my co-leaders and I) view certain moves and give corresponding instructions and the dancers' anatomy or even their differing interpretations of certain instructions.

MoveMatch intends to give objective numerical evaluations that standardize each dancer to the same reference dance video, allowing each individual to have their tailored practice. Ultimately, the project hopes to reduce the instructors' workload to bringing the group performance together, increasing the effectiveness group practices.

## Demo
The user can:
* Choose a short dance reference.
* Practice a specific section.
* Record an attempt.
* Watch their attempt alongside the reference.
* Receive understandable feedback.
* Retry and compare results.
Video pending!

## Screenshots
![Alt Text](assets/screenshot_1.png)
![Alt Text](assets/screenshot_2.png)
![Alt Text](assets/screenshot_3.png)
![Alt Text](assets/screenshot_4.png)
![Alt Text](assets/screenshot_5.png)
![Alt Text](assets/screenshot_6.png)

## Contributions
* Pose-tracking: Done with MediaPipe Pose Landmarker Lite model. 
* Evaluations: Metrics for evaluation were defined by the developer. The implementation and testing was done with Claude under supervision and quality check of the developer.
* Web framework: Flask routing and backend handling + Jinja2 templating with HTML, CSS, Js
* Database: Done on SQLite, integrated with Flask using Flask-SQLAlchemy
* Display: Video rendering and annotating was done with OpenCV

## Technical decisions
* Due to limited computational resources, the functionality is limited to single-dancer use

# Future work
I initially came across HybrIK: Hybrid analytical-neural inverse kinematics for body mesh recovery, but I couldn't install the repo for local testing on Mac. 3D body mesh promises even more precise sample(cannot install all required setup on Mac)
