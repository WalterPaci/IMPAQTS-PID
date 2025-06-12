## Dataset Overview

**IMPAQTS-PID** provides instances of sentences containing implicit meaning, alongside fine- and coarse-grained annotations, topic clustering information, and multiple-choice questions designed to probe models' interpretive skills.

Each row in the dataset corresponds to a speech excerpt with implicit content. The dataset includes the following:


| Column Name                | Description |
|---------------------------|-------------|
| **IMPAQTS_id**            | The identification number of the source sentence in the original IMPAQTS corpus. |
| **IMPAQTS_file**          | The file name (document ID) in the IMPAQTS corpus from which the sentence was extracted. |
| **IMPAQTS_Implicit_content_tag** | The fine-grained annotation of the type of implicit content, as recorded in the IMPAQTS corpus. |
| **tag_simpl_IMPAQTS-PID** | A coarse-grained version of the implicit content tag, used for higher-level grouping in this dataset. |
| **topic**                 | An integer ID corresponding to the output of a topic modeling procedure applied to the dataset, used to group similar sentences and create coherent multiple-choice options. |
| **text**                  | The sentence or excerpt containing implicit content. |
| **right_answer**          | A cleaned explanation of the implicit content, without explicit reference to its type (as described in the paper). |
| **right_answer_id**       | The letter (e.g., 'A', 'B', etc.) corresponding to the correct answer in the multiple-choice format, useful for evaluation purposes. |
| **MCs**                   | The list of possible answers (including the correct one), formatted for the multiple-choice generation (MCG) task as outlined in the accompanying publication. |

---