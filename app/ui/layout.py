from __future__ import annotations

import gradio as gr

from app.ui import events


TABLE_HEADERS = ["index", "文件名", "状态", "Tag 状态", "NL 状态", "已保存", "错误"]


def build_app() -> gr.Blocks:
    with gr.Blocks(title="CCBTag") as app:
        records_state = gr.State([])
        current_index = gr.State(0)

        gr.Markdown("# CCBTag 图片 Caption 编辑器")
        status = gr.Textbox(label="状态", interactive=False)

        with gr.Row():
            dataset_path = gr.Textbox(label="图片文件夹路径", placeholder="/path/to/dataset", scale=4)
            metadata_location = gr.Radio(
                label="元数据保存位置",
                choices=["caption_json", "同目录"],
                value="caption_json",
                scale=1,
            )
            open_button = gr.Button("打开文件夹", variant="primary")
            save_current_button = gr.Button("保存当前")
            save_all_button = gr.Button("保存全部")

        with gr.Row():
            batch_tag_button = gr.Button("批量生成 Tag")
            batch_nl_button = gr.Button("批量生成 NL")
            batch_both_button = gr.Button("批量生成 Tag + NL")
            skip_edited = gr.Checkbox(label="跳过已人工编辑图片", value=True)

        with gr.Row():
            with gr.Column(scale=2):
                image_table = gr.Dataframe(
                    headers=TABLE_HEADERS,
                    datatype=["str"] * len(TABLE_HEADERS),
                    label="图片列表",
                    interactive=False,
                    wrap=True,
                )
                prev_button = gr.Button("上一张")
                next_button = gr.Button("下一张")

            with gr.Column(scale=3):
                preview_image = gr.Image(label="图片预览", type="filepath", height=560)

            with gr.Column(scale=4):
                tag_model = gr.Dropdown(
                    label="Tag 模型",
                    choices=events.tag_model_choices(),
                    value=events.tag_model_choices()[0] if events.tag_model_choices() else None,
                )
                with gr.Row():
                    generate_tag_button = gr.Button("生成 Tag")
                    apply_rules_button = gr.Button("应用黑名单/替换规则")
                    clear_tags_button = gr.Button("清空 Tag")
                tags_text = gr.Textbox(label="Tags", lines=7)

                nl_model = gr.Dropdown(
                    label="描述模型",
                    choices=events.nl_model_choices(),
                    value=events.nl_model_choices()[0] if events.nl_model_choices() else None,
                )
                with gr.Accordion("NL 服务设置", open=False):
                    nl_endpoint = gr.Textbox(
                        label="NL 服务地址",
                        value="http://127.0.0.1:8000/v1/chat/completions",
                    )
                    nl_model_name = gr.Textbox(label="NL 模型名", value="Minthy/ToriiGate-0.5")
                    nl_api_key = gr.Textbox(label="API Key，可留空", type="password")
                with gr.Row():
                    generate_nl_button = gr.Button("生成 NL")
                    generate_both_button = gr.Button("生成 Tag + NL")
                    clear_nl_button = gr.Button("清空 NL")
                nl_text = gr.Textbox(label="Natural Language Caption", lines=7)

                final_caption = gr.Textbox(label="Final Caption Preview", lines=5, interactive=False)

        with gr.Accordion("批量编辑 Tags", open=False):
            with gr.Row():
                delete_tag_text = gr.Textbox(label="批量删除 Tag，逗号分隔")
                delete_tag_button = gr.Button("批量删除")
            with gr.Row():
                replace_old = gr.Textbox(label="替换前")
                replace_new = gr.Textbox(label="替换后")
                replace_tag_button = gr.Button("批量替换")
            with gr.Row():
                add_tag_text = gr.Textbox(label="批量添加 Tag，逗号分隔")
                prepend_tag = gr.Checkbox(label="前置添加", value=False)
                add_tag_button = gr.Button("批量添加")

        open_outputs = [
            records_state,
            image_table,
            preview_image,
            tags_text,
            nl_text,
            final_caption,
            current_index,
            status,
        ]
        record_outputs = [preview_image, tags_text, nl_text, final_caption, current_index, status]
        edit_outputs = [tags_text, final_caption]

        open_button.click(
            events.open_folder,
            inputs=[dataset_path, metadata_location],
            outputs=open_outputs,
        )
        image_table.select(
            events.select_record,
            inputs=[records_state],
            outputs=record_outputs,
        )
        prev_button.click(
            events.previous_record,
            inputs=[records_state, current_index],
            outputs=record_outputs,
        )
        next_button.click(
            events.next_record,
            inputs=[records_state, current_index],
            outputs=record_outputs,
        )
        tags_text.change(events.preview_caption, inputs=[tags_text, nl_text], outputs=final_caption)
        nl_text.change(events.preview_caption, inputs=[tags_text, nl_text], outputs=final_caption)
        apply_rules_button.click(events.apply_rules_to_current, inputs=[tags_text, nl_text], outputs=edit_outputs)
        clear_tags_button.click(events.clear_tags, inputs=[tags_text, nl_text], outputs=edit_outputs)
        clear_nl_button.click(events.clear_nl, inputs=[tags_text, nl_text], outputs=[nl_text, final_caption])

        generate_tag_button.click(
            events.generate_tag,
            inputs=[records_state, current_index, tag_model, tags_text, nl_text],
            outputs=open_outputs,
        )
        generate_nl_button.click(
            events.generate_nl,
            inputs=[records_state, current_index, nl_model, nl_endpoint, nl_model_name, nl_api_key, tags_text, nl_text],
            outputs=open_outputs,
        )
        generate_both_button.click(
            events.generate_tag_and_nl,
            inputs=[
                records_state,
                current_index,
                tag_model,
                nl_model,
                nl_endpoint,
                nl_model_name,
                nl_api_key,
                tags_text,
                nl_text,
            ],
            outputs=open_outputs,
        )
        save_current_button.click(
            events.save_current,
            inputs=[records_state, current_index, tags_text, nl_text, metadata_location],
            outputs=open_outputs,
        )
        save_all_button.click(
            events.save_all,
            inputs=[records_state, current_index, tags_text, nl_text, metadata_location],
            outputs=open_outputs,
        )

        batch_tag_button.click(
            events.batch_generate_tags,
            inputs=[records_state, tag_model, skip_edited],
            outputs=open_outputs,
        )
        batch_nl_button.click(
            events.batch_generate_nl,
            inputs=[records_state, nl_model, nl_endpoint, nl_model_name, nl_api_key, skip_edited],
            outputs=open_outputs,
        )
        batch_both_button.click(
            events.batch_generate_both,
            inputs=[records_state, tag_model, nl_model, nl_endpoint, nl_model_name, nl_api_key, skip_edited],
            outputs=open_outputs,
        )
        delete_tag_button.click(events.batch_delete_tag, inputs=[records_state, delete_tag_text], outputs=open_outputs)
        replace_tag_button.click(
            events.batch_replace_tag,
            inputs=[records_state, replace_old, replace_new],
            outputs=open_outputs,
        )
        add_tag_button.click(
            events.batch_add_tag,
            inputs=[records_state, add_tag_text, prepend_tag],
            outputs=open_outputs,
        )

    return app
