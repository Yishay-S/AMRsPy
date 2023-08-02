# AMRsPy
A repository that houses the code and the data related to the work:<br/>
<b>A Column Generation Approach for Public Transit Enhanced Robotic Delivery Services</b>

## Authors

- [Yishay Shapira](mailto:eshay7777@gmail.com)
- [Mor Kaspi](mailto:morkaspi@tauex.tau.ac.il)

1. resources
   1. data <br/>
   The instance name order is in the following manner: \<requests\>\_\<robots\>\_\<depots\>\_\<service_lines\>\_\<transfers\><br/>
   The instances are stored in pickle files as a Map class, for each instance there is also a text file with the description of the instance.<br/>
   In order to read a Map class please go to logic/read_map.py for example.<br/>
   For the TLV case study instances there are additional CSV files with nodes and arcs information.
