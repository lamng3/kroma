### NCIT Query Templates
class ncit_query_templates:
    query_children_template = """
    prefix ncit:  <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#>
    prefix rdfs:  <http://www.w3.org/2000/01/rdf-schema#>
    prefix owl:  <http://www.w3.org/2002/07/owl#>

    select ?label ?rel_label ?obj_label
    from <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.rdf>
    where {
      VALUES ?superclass { ncit:<code> }
      ?subject rdfs:subClassOf+ ?superclass ;
               rdfs:label ?label .
      OPTIONAL {
          ?subject ?relation ?object . 
          ?relation a owl:ObjectProperty ; 
                    rdfs:label ?rel_label . 
          ?object rdfs:label ?obj_label .
      }
    }
    order by asc(?label)  limit 500
    """

    query_parents_template = """
    prefix ncit:  <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#>
    prefix rdfs:  <http://www.w3.org/2000/01/rdf-schema#>
    prefix owl:  <http://www.w3.org/2002/07/owl#>

    select ?label ?rel_label ?obj_label
    from <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.rdf>
    where {
      VALUES ?subclass { ncit:<code> }
      ?subclass rdfs:subClassOf+ ?subject ;
               rdfs:label ?label .
      OPTIONAL {
          ?subject ?relation ?object . 
          ?relation a owl:ObjectProperty ; 
                    rdfs:label ?rel_label . 
          ?object rdfs:label ?obj_label .
      }
    }
    order by asc(?label)  limit 500
    """

    query_synonyms_template = """
    prefix ncit: <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#>
    prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    prefix owl: <http://www.w3.org/2002/07/owl#>

    select DISTINCT ?subject ?prop_label ?prop_value ?annotations
    from <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.rdf>
    where {
      values ?subject { ncit:<code> }
      ?subject ?prop ?prop_value .
      ?prop rdfs:label ?prop_label ;
        a owl:AnnotationProperty .

      optional {
        SELECT DISTINCT ?subject ?prop ?prop_value 
               (GROUP_CONCAT(?annot ; separator=' | ') AS ?annotations)
        where {
          ?an a owl:Axiom ;
             owl:annotatedSource ?subject ;
             owl:annotatedProperty ?prop ;
             owl:annotatedTarget ?prop_value ;
                ?annotation ?annot_value .
          ?annotation rdfs:label ?annotation_label .
          BIND( CONCAT( ?annotation_label , '=' , ?annot_value ) AS ?annot )
          } GROUP BY ?subject ?an ?prop ?prop_value
       }
    }
    order by ?prop_label
    limit 100
    """

    query_label_template = """
    PREFIX ncit:  <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#>
    PREFIX rdfs:  <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?label
    FROM <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.rdf>
    WHERE {
        ncit:<code> rdfs:label ?label .
    }
    """

### DOID Query Templates
class doid_query_templates:
    query_synonyms_template = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX obo: <http://purl.obolibrary.org/obo/>
    PREFIX oboInOwl: <http://www.geneontology.org/formats/oboInOwl#>

    SELECT ?id ?label ?exact_synonym ?syn_is_acronym
    WHERE {
      ?class a owl:Class ;
        oboInOwl:hasOBONamespace "disease_ontology" ;
        oboInOwl:id ?id ;
        rdfs:label ?label ;
        oboInOwl:hasExactSynonym ?exact_synonym .

      BIND(
        EXISTS {
          [] a owl:Axiom ;
            owl:annotatedSource ?class ;
            owl:annotatedProperty oboInOwl:hasExactSynonym ;
            owl:annotatedTarget ?exact_synonym ;
            oboInOwl:hasSynonymType obo:OMO_0003012 .
        } AS ?syn_is_acronym
      )

      FILTER NOT EXISTS { ?class owl:deprecated true }
      FILTER (?id = "DOID:<code>")
    }
    ORDER BY ?id ?exact_synonym
    LIMIT 20
    """

    query_parents_template = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX oboInOwl: <http://www.geneontology.org/formats/oboInOwl#>
    PREFIX DOID: <http://purl.obolibrary.org/obo/DOID_>

    SELECT ?id ?label ?xref
    WHERE {
        ?class oboInOwl:id ?id ;
            rdfs:label ?label ;
            oboInOwl:hasDbXref ?xref ;
            ^rdfs:subClassOf* DOID:<code> .
        FILTER( REGEX(?xref, "^(ICD9|ICD10|SNOMED)") )
        FILTER NOT EXISTS { ?class owl:deprecated true }
    }
    LIMIT 20
    """

    query_children_template = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX oboInOwl: <http://www.geneontology.org/formats/oboInOwl#>
    PREFIX DOID: <http://purl.obolibrary.org/obo/DOID_>

    SELECT ?id ?label ?xref
    WHERE {
        ?class oboInOwl:id ?id ;
            rdfs:label ?label ;
            oboInOwl:hasDbXref ?xref ;
            rdfs:subClassOf* DOID:<code> .
        FILTER( REGEX(?xref, "^(ICD9|ICD10|SNOMED)") )
        FILTER NOT EXISTS { ?class owl:deprecated true }
    }
    LIMIT 20
    """

    query_label_template = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX DOID: <http://purl.obolibrary.org/obo/DOID_>

    SELECT ?label
    WHERE {
        DOID:<code> rdfs:label ?label .
    }
    """

### DBpedia Query Templates
class dbpedia_query_templates:
    query_parents_template = """
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX dbo: <http://dbpedia.org/ontology/>

    SELECT ?object WHERE{ dbo:<code> rdf:type ?object }
    LIMIT 10
    """

    query_children_template = """
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX dbo: <http://dbpedia.org/ontology/>

    SELECT ?object WHERE{ ?object rdf:type dbo:<code> . }
    LIMIT 10
    """

    query_synonyms_template = """
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

    SELECT DISTINCT ?object ?synonym
    WHERE {
      ?object rdf:type dbo:<code> .
      ?object skos:altLabel ?synonym .
      FILTER (LANG(?synonym) = "en")
    }
    LIMIT 10
    """

    query_label_template = """
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

    SELECT DISTINCT ?altLabel
    WHERE {
      dbo:<code> skos:altLabel ?altLabel .
      FILTER (LANG(?altLabel) = "en")
    }
    LIMIT 10
    """