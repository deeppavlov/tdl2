![TDL logo](/banner3.gif)

This is a GitHub page of the 2nd part of Theoretical Deep Learning course held by Neural Networks and Deep Learning Lab., MIPT. For the first part, see [this page](https://github.com/deepmipt/tdl).
**Note that two parts are mostly mutually independent.**

The working language of this course is Russian.

**Location:** Moscow Institute of Physics and Technology, ФИЗТЕХ.ЦИФРА building, room 5-22.

**Time:** Monday, 10:45. The first lecture (on September ~9~ 16) will start at ~11:00~ 10:45.

**Videos** will be added to [this](https://www.youtube.com/playlist?list=PLt1IfGj6-_-eiAGKvcZrHCp1mejmxMCiX) playlist.

Lecture slides, homework assignments and videos will appear in this repo and will be available for everyone. However, we can guarantee that we will check your homework only if you are a MIPT student.

Further announcements will be in our Telegram chat: https://t.me/joinchat/D_ljjxJHIrD8IuFvfqVLPw

## Syllabus:

This syllabus is not final and may change.

1. **16.09.19** Introduction. Short recap of TDL#1. Course structure. Organization notes. [Slides](/slides/Intro.pdf), [video](https://youtu.be/xwfAiaJ74Vk).

2. **16.09.19** Worst-case generalization bounds: growth function, VC-dimension. [Slides](/slides/Worst_case_bounds.pdf) (up to page 10), [video](https://youtu.be/fzKGRxk4DXk).

3. **23.09.19** Worst-case generalization bounds: margin loss, fat-shattering dimension, covering numbers. [Slides](/slides/Worst_case_bounds.pdf) (up to page 10, the same as in the previous lecture), [video](https://youtu.be/qheV9dDyLcg).

4. **30.09.19** Worst-case generalization bounds: McDiarmid inequality, Rademacher complexity,  spectral complexity. [Slides](/slides/Worst_case_bounds.pdf) (pages 11-17), [video](https://youtu.be/4Q3zoMTBamc).

5. **07.10.19** Worst-case generalization bounds: bound for deep ReLU nets. [Slides](/slides/Worst_case_bounds.pdf) (pages 11-17), [video](https://youtu.be/8MuJM4S3UyM).

6. **14.10.19** Worst-case generalization bounds: failure of uniform bounds. [Slides](/slides/Worst_case_bounds.pdf) (pages starting from 18). PAC-bayesian bounds: at most countable hypothesis classes. [Slides](/slides/PAC_bayesian_bounds.pdf) (up to page 3). [Video](https://youtu.be/V-yhl7usGkU).

7. **21.10.19** PAC-bayesian bounds: uncountable hypothesis classes. [Slides](/slides/PAC_bayesian_bounds.pdf) (pages 3-6). [Video](https://youtu.be/7rFIVhLXflQ).

8. **28.10.19** PAC-bayesian bounds: dealing with stochasticity requirement, margin-based bound for deep ReLU nets. [Slides](/slides/PAC_bayesian_bounds.pdf) (pages 7-19). [Video](https://youtu.be/8x4RqMRRsCM).

9. **11.11.19** PAC-bayesian bounds: margin-based bound for deep ReLU nets. [Slides](/slides/PAC_bayesian_bounds.pdf) (pages 16-20). [Video](https://youtu.be/2xKmJuDnpLw).

10. **18.11.19** Compression-based bounds: re-deriving a margin-based bound for deep ReLU nets. [Slides](/slides/PAC_bayesian_bounds.pdf) (pages 18-26). [Video](https://youtu.be/zkx3F1XlMfU).

11. **2.12.19 (?)** Implicit bias of gradient descent.

## Prerequisites:

* Basic calculus / probability / linear algebra
* Labs are given as jupyter notebooks 
* We use python3; need familiriaty with numpy, pytorch, matplotlib
* Some experience in DL (not the first time of learning MNIST)
* Labs are possible to do on CPU, but it can take quite a long time to train (~1-2 days).
    
## Grading:

This course will contain 1 lab and 2 theoretical assignments. 
There also will be an oral exam (in the form of interview) at the end of the course.

Let p_{hw} = "your points for homeworks" / "total possible points for homeworks (excluding extra points)". Define p_{exam} analogously.

Your final grade will be computed as follows:
grade = min(10, p_{hw} * k_{hw} + p_{exam} * k_{exam}), where the coefficents are:
* k_{hw} = 4
* k_{exam} = 6

This numbers are not final and can change.

Send your homeworks to tdl_course_mipt@protonmail.com

E-mails should be named as "Lab or theory" + "number" + "-" + "Your Name and Surname"

## Homeworks:

~[The 1st theoretical assignment](/hw_theory/tdl2_theory1.pdf) is out! Deadline: 4.11.2019, 12:00 Moscow time.~

~[The 2nd theoretical assignment](/hw_theory/tdl2_theory2.pdf) is out! Deadline: 2.12.2019, 12:00 Moscow time.~

## Exam/zachet:

Syllabus is [here](tdl2_exam_syllabus.pdf).
Grading and question balance between main and auxiliary parts is not final and may change.
**Start with digging into main part first.**

**WARNING:** 
If you have this course for zachet, and you want to have your zachet in time (i.e., in a zachet week), the only available option is December 16.

## Course staff:

- [Eugene Golikov](https://github.com/varenick) - course admin, lectures, homeworks
- [Ivan Skorokhodov](https://github.com/universome) - homeworks

This course is dedicated to the memory of [Maksim Kretov](https://github.com/kretovmk) | 30.12.1986 - 13.02.2019, without whom this course would have never been created.
