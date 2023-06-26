class TestConstants1:
    def __init__(self):
        self.actin_concentration = 1000
        self.arp23_concentration = 20
        self.cp_concentration = 10

 
        self.kp_on = 0.5E-2
        self.kp_off = 0.5
        self.arp23_on = 2.5E-2
        self.cp_on = 2.5E-3

                
        self.monomer_diameter = 0.1
        self.cp_diameter=0.1
        
        self.tirf_distance_height = 1.0

        self.interface_diff = 0.01
        self.interface_drift_scale = -1
class TestConstants2:
    def __init__(self):
        self.actin_concentration = 1000
        self.arp23_concentration = 1
        self.cp_concentration = 10

 
        self.kp_on = 0.5E-2
        self.kp_off = 0.5
        self.arp23_on = 2.5E-4
        self.cp_on = 2.5E-2

                
        self.monomer_diameter = 0.1
        self.cp_diameter=0.7
        self.tirf_distance_height = 1.0


        self.interface_diff = 0.4
        self.interface_drift_scale = -1





class LiteratureConstants:
    def __init__(self):
        #All concentrations are in units of micromolars
        self.actin_concentration = 5
        self.arp23_concentration = 0.1
        self.cp_concentration = 0.1

 
        #All rates are in units of 1/(micromolars * seconds)
        self.kp_on = 11.6
        self.kp_off = 1.4
        self.arp23_on = 0.003 #* 50
        self.cp_on = 2.6

        # All sizes are in units of um
        self.monomer_diameter=2.7E-3
        self.cp_diameter=2.7E-3
        self.tirf_distance_height = 0.1

        #Units in micrometer^2/s
        self.interface_diff = 2.07E-3

        #Units of micrometer/second
        self.interface_drift_scale = -0.75 #(-f/gamma)



