# 1th week
## total task
  1. reproduce rednet
  2. realize quantitative training
  3. realize data enhancement

## active time
7.16+19:00 - 7.19+9:00

62 hours totally

sleeping for 5h per day, 3h for trivia per day

44h left

## steps by steps

(1) **use resnet-50 to train vww in 3090** </br>
    because rednet derive from it. we need a benchmark.</br>
    the lightest rednet for cls is rednet-26, we choose this version for saving time. </br>

(2) **realize quantitative training in resnet-26(conv) in 2070** </br>
    at last, we should do quantitative training in rednet-26, but now we need a guaranteed outcome for pre

(3) **realize data enhancement and give a outcome based on resnet-26 in 2070**</br>
    same reason as above, we need a guaranteed outcome for pre

(4) do 

hope I can finish steps above before the sun rises in 7-17</br>
we'll get 4 results:</br>
resnet-alone(3090); resnet-quantitative training(2070); resnet-data enhancement(2070); resnet-both(3090)

(4) **reproduce rednet-26, use it to train vww in 3090, and compare the results with that of step(1)**</br>
    this step may cost the most time, but things go easier after finish this step

hope I can finish this step before the sun goes down in 7-17

(5) **realize quantitative training in rednet-26(inv) in 2070**

(6) **realize data enhancement and give a outcome based on rednet-26(inv) in 2070**</br>

hope I can finish this step before the sun rises in 7-18
up to this time, we'll have results:</br>
rednet-alone(3090); rednet-quantitative(2070);  rednet-data enhancement(2070); rednet-both(3090); </br>


**at last, in the pre, I should present**:</br>
resnet-alone--rednet-alone(3090); resnet-both--rednet-both(3090);</br>
reset-alone--resnet-both(3090); rednet-alone--rednet-both(3090)</br>
resnet-alone--resnet-q training(2070); rednet-alone--rednet-q training(2070)</br>
resnet-alone--resnet-DE(2070); rednet-alone--rednet-DE(2070)</br>


    




