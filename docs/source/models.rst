.. _Models:

Available Models
----------------

#. XENONnT_O1_rate: 

   This model was trained using synthetic data of the expected total
   rate in xenonNT for dark matter particles with :math:`\mathcal{O}_{1}`
   with :math:`M_{dm} \in []`, :math:`\sigma \in []` 
   and :math:`\theta \in []`.
   In order to analyse a new data :math:`x` it must be a np.array with shape (1). CHECK THIS

#. XENONnT_O1_drate: 

   This model was trained using synthetic data of the expected differential
   rate in xenonNT for dark matter particles with :math:`\mathcal{O}_{1}`
   with :math:`M_{dm} \in []`, :math:`\sigma \in []` 
   and :math:`\theta \in []`.
   In order to analyse a new data :math:`x` it must be a np.array with shape (1,20). CHECK THIS

#. XENONnT_O1_s1s2: 

   This model was trained using synthetic data of the expected s1s2
   data in xenonNT for dark matter particles with :math:`\mathcal{O}_{1}`
   with :math:`M_{dm} \in []`, :math:`\sigma \in []` 
   and :math:`\theta \in []`.
   In order to analyse a new data :math:`x` it must be a np.array with shape (1,65,65). CHECK THIS
