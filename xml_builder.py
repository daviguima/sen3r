xml_string = """
<graph id="someGraphId">
    <version>1.0</version>
    <node id="regionalSubset">
      <operator>Subset</operator>
      <sources>
        <source>${source}</source>
      </sources>
      <parameters>
        <geoRegion>%s</geoRegion>
        <copyMetadata>true</copyMetadata>
      </parameters>
    </node>
    <node id="importShapefile">
      <operator>Import-Vector</operator>
      <sources>
        <source>regionalSubset</source>
      </sources>
      <parameters>
        <vectorFile>%s</vectorFile>
        <separateShapes>false</separateShapes>
      </parameters>
    </node>
    <node id="maskArea">
      <operator>Land-Sea-Mask</operator>
      <sources>
        <source>importShapefile</source>
      </sources>
      <parameters>
        <landMask>false</landMask>
        <useSRTM>false</useSRTM>
        <geometry>%s</geometry> 
        <invertGeometry>false</invertGeometry>
        <shorelineExtension>0</shorelineExtension>
      </parameters>
    </node>
  </graph>
"""