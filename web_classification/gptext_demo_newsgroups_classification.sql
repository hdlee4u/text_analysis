----------------------------------------------------------------------------------------------------------------------
--2) GPText for search and SVM prediction
----------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------------------------------
--					ETL				          --
----------------------------------------------------------------------------------	
        --Load the 20 News group dataset (for GPText use case)
	drop table if exists gpdemo.twenty_news_groups cascade;
	create table gpdemo.twenty_news_groups 
	(DOC_ID bigint, CONTENTS text, label text) 
	distributed by (DOC_ID);
	copy gpdemo.twenty_news_groups from '/data/vatsan/uci_datasets/20_news_groups_cleansed.tsv' delimiter E'\t' header;

	--Display some rows from the table
	select * from gpdemo.twenty_news_groups limit 10;

	--Label distribution
	--Stats: There are 20 labels, 1000 instances for each label except one which has 997. So distribution is even.
	select label, count(doc_id)
	from gpdemo.twenty_news_groups
	group by label
	order by label
	
	

----------------------------------------------------------------------------------
--					ANALYTICS				  --
----------------------------------------------------------------------------------	

--   i) Create Index
        --preprocess:
        --drop any existing index
        select gptext.drop_index('gpdemo','twenty_news_groups');
        --a) Create Empty Index
	select * from gptext.create_index('gpdemo','twenty_news_groups','doc_id','contents');
        --b) Enable terms table
        select gptext.enable_terms('vatsandb.gpdemo.twenty_news_groups','contents');	        
        --c) Populate Index
	select * from gptext.index(TABLE(select * from gpdemo.twenty_news_groups), 'vatsandb.gpdemo.twenty_news_groups');
        --d) Commit Index
	select * from gptext.commit_index('vatsandb.gpdemo.twenty_news_groups');

	-- Show sample search
	select * from gptext.search(TABLE(select * from gpdemo.twenty_news_groups),
				     'vatsandb.gpdemo.twenty_news_groups',
				     'mac',
				     null
				    );
-- ii) Create Terms Table
	drop table if exists gpdemo.twenty_news_groups_terms cascade;
	create table gpdemo.twenty_news_groups_terms as
	(
		select *
		from gptext.terms(
				    TABLE(select * from gpdemo.twenty_news_groups),
				    'vatsandb.gpdemo.twenty_news_groups',
				    'contents',
				    '*:*',
				    null
				  )
	) distributed by (id);

-- iii) Create term dictionary

        drop table if exists gpdemo.twenty_news_groups_terms_dict cascade;
	create table gpdemo.twenty_news_groups_terms_dict as
	(
		select row_number() over(order by term asc) as idx,
		       term, 
		       term_freq
		from(
			select term,
			       count(*) as term_freq
			from gpdemo.twenty_news_groups_terms
			where term is not null
			group by term
		) q where term_freq > 50 -- disregard terms which occur less than 50 times atleast (long tail).
		
	) distributed by (idx);       
	
--  iv) Create sparse vectors representation using tf-idf scores for the corpus
	drop table if exists gpdemo.twenty_news_groups_corpus cascade;
	create table gpdemo.twenty_news_groups_corpus as (
		select 
		     terms.id, 
		     gptext.gen_float8_vec(
			array_agg(dict.idx),                        -- Array of indices
			array_agg(array_upper(terms.positions, 1)), -- Array of values at indices
			gptext.count_t('gpdemo.twenty_news_groups_terms_dict') --# terms in dictionary
		     )::madlib.svec sfv
		from
		     gpdemo.twenty_news_groups_terms terms,
		     gpdemo.twenty_news_groups_terms_dict dict
		where
		     terms.term = dict.term
		group by terms.id
	) distributed by (id);

        -- TF-IDF feature vect
        drop table if exists gpdemo.twenty_news_groups_fvect_tfidf;
	create table gpdemo.twenty_news_groups_fvect_tfidf as (
	   select
		id as doc_id,
		madlib.svec_mult( sfv, logidf ) as tf_idf
	    from gpdemo.twenty_news_groups_corpus ,
		 ( 
		    select madlib.svec_log(
			  	madlib.svec_div(
			  	    count(sfv)::madlib.svec,
			  	    madlib.svec_count_nonzero(sfv)
			  	)
			    ) as logidf 
		    from gpdemo.twenty_news_groups_corpus 
		 ) foo
	    order by doc_id
	) distributed by (doc_id);

        -- Display the corpus
        select doc_id, ;
        
--   v) Create Train & Test sets, to build a predictive model using SVM
	drop table if exists gpdemo.twenty_news_groups_fvect_tfidf_all cascade;
	create table gpdemo.twenty_news_groups_fvect_tfidf_all as (
		select t1.doc_id as id, 
		       madlib.svec_return_array(t1.tf_idf) as ind,
		       random() as rand_num,
		       -- We'll detect if a news group discussion is about computers or not
		       case when t2.label in ('comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','comp.windows.x') 
			    then 1 else -1 
		       end as label
		from gpdemo.twenty_news_groups_fvect_tfidf t1, gpdemo.twenty_news_groups t2
		where t1.doc_id = t2.doc_id 
		      and label in (
		      -- Picking 10 labels - 5 about computers, 5 about a mix of politics & religion - ensuring distribution is still even
			'talk.politics.guns',
			'talk.politics.mideast',
			'talk.politics.misc',
			'talk.religion.misc',
			'alt.atheism',
			'comp.graphics',
			'comp.os.ms-windows.misc',
			'comp.sys.ibm.pc.hardware',
			'comp.sys.mac.hardware',
			'comp.windows.x'
		)
	) distributed by (id);

	-- Training set
	drop table if exists gpdemo.twenty_news_groups_fvect_tfidf_train cascade;
	create table gpdemo.twenty_news_groups_fvect_tfidf_train as (
		select * from gpdemo.twenty_news_groups_fvect_tfidf_all
		where rand_num < 0.60
	) distributed by (id);

	-- Test set
	drop table if exists gpdemo.twenty_news_groups_fvect_tfidf_test cascade;
	create table gpdemo.twenty_news_groups_fvect_tfidf_test as (
		select * from gpdemo.twenty_news_groups_fvect_tfidf_all
		where rand_num > 0.60
	) distributed by (id);	
	
       -- Linear SVM Classifier training
       drop table if exists gpdemo.svm_mdl_newsgrps cascade ;
       drop table if exists gpdemo.svm_mdl_newsgrps_param cascade ;
       select madlib.lsvm_classification('gpdemo.twenty_news_groups_fvect_tfidf_train',
					  'gpdemo.svm_mdl_newsgrps',
					  false
					 );
	-- Linear SVM prediction
	drop table if exists gpdemo.twenty_news_groups_svm_predictions cascade;
	select madlib.lsvm_predict_batch(
					  'gpdemo.twenty_news_groups_fvect_tfidf_test',
					  'ind',
					  'id',
					  'gpdemo.svm_mdl_newsgrps',
					  'gpdemo.twenty_news_groups_svm_predictions',
					  false
					 );

       -- Show SVM prediction accuracy
       -- Stats: 90.20% precision, 15.91% recall
       select 'svm'::text as model, 
	      sum(true_positive)*100.0/(sum(true_positive)+sum(false_positive)) as precision,
	      sum(true_positive)*100.0/(sum(true_positive)+sum(false_negative)) as recall
       from
       (       -- Verify if the predicted label is equal to the actual label	
	       select id, 	
		      case when actual_label=predicted_label and actual_label=1 then 1 else 0 end as true_positive,
		      case when actual_label=-1 and predicted_label=1 then 1 else 0 end as false_positive,
		      case when actual_label=1 and predicted_label=-1 then 1 else 0 end as false_negative
	       from
	       (
	               -- Show id, actual label & predicted label				  
		       select t1.id, 
			      t1.label as actual_label,
			      case when t2.prediction > 0 then 1 else -1 end as predicted_label 
			      
		       from gpdemo.twenty_news_groups_fvect_tfidf_test t1, gpdemo.twenty_news_groups_svm_predictions t2
		       where t1.id = t2.id
	       ) q1
       ) q2;

----------------------------------------------------------------------------------------------------------------------


