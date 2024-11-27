## Basic git commands

*	Cloning the repository to your local environment - git clone https://github.com/fazlur97/Python_EDA_Group.git
*	Before making a change in your local environment, make sure to pull in the latest changes (Very important to avoid merge conflicts) – git pull
*	After making changes in your local environment
    *	Add the files you want to merge to the staging area 
        * To add each file - git add \<filenames\>
        * To add all the files – git add .
        * To delete file - git rm \<filenames\>
    * Commit the changes in the staging area – git commit -m “Write a commit message”
    * Push your changes to the github remote repository – git push
*	To view the files that have been changed in your local repository – git status
*	To view the changes you have made in each file – git diff \<filenames\>
*   To view what file you have changed - git diff --name-only
*	To view your latest commits – git log
*	If you want to revert your changes 
    *	Restore changes in staging area - git restore --staged \<filenames\>
    *	Restore changes that’s not staged - git restore \<filenames\>
    *	To remove all your local changes – git stash


* Additional Info
* If you haven't make a token access for general usage
* Go to github -> Click your profile picure on top right → Setting
* Developer Setting → Personal Access Token → Tokens (classic)
* Generate new token and tick everything, you will got token looks like this ghp_0qSOi5EwBJjm2Ba0vV87UaNPEUUxus2
* Windows User :
    * Go to Control Panel->User Accounts->Credential Manager
    * Add a generic credential → Internet address will be "git:https://github.com" and you need to type in your username and password will be your GitHub Personal Access Token → Click Ok and you are done