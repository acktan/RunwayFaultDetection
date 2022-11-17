[![pipeline status](https://gitlab.code.hfactory.io/adrian.tan/colasproject/badges/main/pipeline.svg)](https://gitlab.code.hfactory.io/adrian.tan/colasproject/-/commits/main)
[![coverage report](https://gitlab.code.hfactory.io/adrian.tan/colasproject/badges/main/coverage.svg)](https://gitlab.code.hfactory.io/adrian.tan/colasproject/-/commits/main)

# Multi-label classification of runway images to determine the different defects they may have 
**Colas x MSc X-HEC Data Science for Business**

## Presentation of the project

### Quick overview of the project 
This computer vision challenge has been proposed to sereval teams from the MSc X-HEC Data Science for Business by the AI / Data Science team of the building group COLAS, subsidiary of the Bouygues Group. COLAS works on several businesses (Road, rail, manufacturing, quarries, pipeline, ...)

### Objective 
Identify asphalt mixed defeact on airport track, and rank them with a clear data visualization. Identify airport with the best business opportunity for Colas, regarding its activities. Extend the consideration to all kind of Colas Business opportunities

### Step for the chalenge
1. Collect french airport open data
The objective of this step is to identify & collect open data of french airports (name, geolocalization...).
We used the BD Carto & BD Orthophoto from the French public organization IGN : https://geoservices.ign.fr/bdcarto

2. Collect french orthophotos
The objective was to collect IGN's orthophotos (each picture 25k x 25k pixels represents a square of 5km per 5km). Again, we found them on the IGN opendata platform : https://geoservices.ign.fr/bdortho#telechargement

3. Extract airport images
Here the objective was to extract picture for each airport. To achieve this objective, we linked the airports geolocalization from step 1 & orthophotos from step 2 to extract airports pictures.

4. Extract runway images
Similarly, the objective was to extract picture for each runaway from each airport. We used the tool [Label Studio](https://labelstud.io/guide/) to label the runaways manually before extracting them with Python

5. Modeling
Here, the objective was to Train a computer vision model to predict damages (multi label image classification). Metrics is weighted F1 score. Shout out to [Ashref Maiza](https://github.com/ashrefm/) who wrote this  really cool [article](https://towardsdatascience.com/multi-label-image-classification-in-tensorflow-2-0-7d4cf8a4bc72) on which we based a large part of our model.

6. Business opportunities
This step was particularly important to the Colas team. We had to identify similar uses cases or new business opportunities for COLAS Group based on the computer vision technology.

7. Pitch
Last but not least, we presented the team's results and especially the potential of the technology for the Colas Group during a presentation in front of several senior COlAS collaborators (Chief Digital Officer, Chief Data Officer, Digital and Core business applications Director, Chief Data Officer, Lead Data/AI, ...) and HEC professors (head of Corporate Partnership Dpt, Prof. Chair Holder of Bouygues Chair “Smart city and the common good”, Associate Dean for Research)

## Structure of the repository




## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.code.hfactory.io/adrian.tan/colasproject.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

- [ ] [Set up project integrations](https://gitlab.code.hfactory.io/adrian.tan/colasproject/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Automatically merge when pipeline succeeds](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing(SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thank you to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README
Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
