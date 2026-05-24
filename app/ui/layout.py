from __future__ import annotations

import gradio as gr

from app.core.preview import image_preview_html
from app.ui import events


TABLE_HEADERS = ["#", "文件名", "状态", "Tag", "NL", "保存", "错误"]


APP_CSS = """
.ccbtag-shell { max-width: 1760px; margin: 0 auto; }
.ccbtag-status textarea { min-height: 42px !important; }
.ccbtag-list table { font-size: 13px; }
.ccbtag-list th, .ccbtag-list td { white-space: nowrap !important; }
.ccbtag-list td:first-child { max-width: 220px; overflow: hidden; text-overflow: ellipsis; }
.ccbtag-preview { display: flex; flex-direction: column; align-items: center; }
.ccbtag-preview img { max-height: calc(100vh - 220px); object-fit: contain; width: 100%; }
.ccbtag-editor textarea { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.ccbtag-actions button { min-height: 42px; }
"""


SHORTCUT_JS = """
function() {
    document.addEventListener('keydown', (e) => {
        // Skip if typing in an input field
        if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
        
        if (e.key === 'a' || e.key === 'A') {
            const btn = document.querySelector('#prev-button');
            if (btn) btn.click();
        }
        if (e.key === 'd' || e.key === 'D') {
            const btn = document.querySelector('#next-button');
            if (btn) btn.click();
        }
        if ((e.ctrlKey || e.metaKey) && (e.key === 's' || e.key === 'S')) {
            e.preventDefault();
            const btn = document.querySelector('#save-button');
            if (btn) btn.click();
        }
    });
}
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="CCBTag") as app:
        records_state = gr.State([])
        current_index = gr.State(0)

        with gr.Column(elem_classes=["ccbtag-shell"]):
            gr.Markdown("# CCBTag 图片 Caption 编辑器")
            status = gr.Textbox(
                label="状态",
                interactive=False,
                visible=False,
                elem_classes=["ccbtag-status"],
            )

            with gr.Tabs():
                with gr.Tab("🖼️ 标注工作台"):
                    with gr.Row():
                        with gr.Column(scale=3, min_width=360, variant="panel"):
                            dataset_path = gr.Textbox(
                                label="图片文件夹路径", placeholder="/path/to/dataset"
                            )
                            with gr.Row():
                                metadata_location = gr.Radio(
                                    label="元数据保存位置",
                                    choices=events.metadata_location_choices(),
                                    value=events.default_metadata_location(),
                                    scale=2,
                                )
                                open_button = gr.Button(
                                    "打开文件夹", variant="primary", scale=1
                                )
                            image_table = gr.Dataframe(
                                headers=TABLE_HEADERS,
                                datatype="str",
                                label="图片列表",
                                interactive=False,
                                wrap=False,
                                elem_classes=["ccbtag-list"],
                            )

                        with gr.Column(
                            scale=5, min_width=420, elem_classes=["ccbtag-preview"]
                        ):
                            preview_image = gr.HTML(
                                label="图片预览", value=image_preview_html(None)
                            )
                            with gr.Row(
                                variant="panel", elem_classes=["ccbtag-actions"]
                            ):
                                prev_button = gr.Button(
                                    "⬅️ 上一张", elem_id="prev-button"
                                )
                                save_current_button = gr.Button(
                                    "💾 保存 (Ctrl+S)",
                                    variant="primary",
                                    elem_id="save-button",
                                )
                                save_all_button = gr.Button("保存全部")
                                next_button = gr.Button(
                                    "下一张 ➡️", elem_id="next-button"
                                )

                        with gr.Column(
                            scale=4,
                            min_width=380,
                            variant="panel",
                            elem_classes=["ccbtag-editor"],
                        ):
                            with gr.Accordion("⚙️ 模型与生成参数", open=False):
                                tag_model = gr.Dropdown(
                                    label="Tag 模型",
                                    choices=events.tag_model_choices(),
                                    value=events.tag_model_choices()[0]
                                    if events.tag_model_choices()
                                    else None,
                                )
                                workbench_tag_categories = gr.CheckboxGroup(
                                    label="Tag 生成保留类别",
                                    choices=events.tag_category_choices(),
                                    value=events.default_kept_tag_categories(),
                                )
                                nl_model = gr.Dropdown(
                                    label="描述模型",
                                    choices=events.nl_model_choices(),
                                    value=events.nl_model_choices()[0]
                                    if events.nl_model_choices()
                                    else None,
                                )
                                with gr.Accordion("NL 服务设置", open=False):
                                    nl_endpoint = gr.Textbox(
                                        label="NL 服务地址",
                                        value=events.default_nl_endpoint(),
                                    )
                                    nl_model_name = gr.Textbox(
                                        label="NL 模型名",
                                        value=events.default_nl_model_name(),
                                    )
                                    nl_api_key = gr.Textbox(
                                        label="API Key，可留空",
                                        type="password",
                                        value=events.default_nl_api_key(),
                                    )
                                    shuffle_tags = gr.Checkbox(
                                        label="生成前打乱 Tag 顺序",
                                        value=events.default_shuffle_tags(),
                                    )
                                    image_resize_mode = gr.Dropdown(
                                        label="图片压缩",
                                        choices=["None", "1MP"],
                                        value=events.default_nl_image_resize_mode(),
                                    )

                            with gr.Group():
                                tags_text = gr.Textbox(label="Tag", lines=5)
                                with gr.Row(elem_classes=["ccbtag-actions"]):
                                    generate_tag_button = gr.Button(
                                        "生成 Tag", variant="secondary", size="sm"
                                    )
                                    apply_rules_button = gr.Button(
                                        "应用规则", size="sm"
                                    )
                                    clear_tags_button = gr.Button("清空 Tag", size="sm")

                            with gr.Group():
                                nl_text = gr.Textbox(label="自然语言描述", lines=5)
                                with gr.Row(elem_classes=["ccbtag-actions"]):
                                    generate_nl_button = gr.Button(
                                        "生成 NL", variant="secondary", size="sm"
                                    )
                                    generate_both_button = gr.Button(
                                        "生成 Tag + NL", variant="primary", size="sm"
                                    )
                                    clear_nl_button = gr.Button("清空 NL", size="sm")

                            final_caption = gr.Textbox(
                                label="最终 Caption 预览", lines=3, interactive=False
                            )

                with gr.Tab("📦 批量处理中心"):
                    with gr.Row():
                        with gr.Column(variant="panel"):
                            gr.Markdown("### 自动批量生成")
                            batch_tag_categories = gr.CheckboxGroup(
                                label="Tag 生成保留类别",
                                choices=events.tag_category_choices(),
                                value=events.default_kept_tag_categories(),
                            )
                            skip_edited = gr.Checkbox(
                                label="跳过已人工编辑图片", value=True
                            )
                            with gr.Row(elem_classes=["ccbtag-actions"]):
                                batch_tag_button = gr.Button("批量生成 Tag")
                                batch_nl_button = gr.Button("批量生成 NL")
                                batch_both_button = gr.Button("批量生成 Tag + NL")
                                stop_batch_button = gr.Button(
                                    "停止批量生成", variant="stop"
                                )

                        with gr.Column(variant="panel"):
                            gr.Markdown("### 批量替换与删除")
                            with gr.Row():
                                delete_tag_text = gr.Textbox(
                                    label="批量删除 Tag，逗号分隔"
                                )
                                delete_tag_button = gr.Button("批量删除")
                            with gr.Row():
                                replace_old = gr.Textbox(label="替换前")
                                replace_new = gr.Textbox(label="替换后")
                                replace_tag_button = gr.Button("批量替换")
                            with gr.Row():
                                add_tag_text = gr.Textbox(
                                    label="批量添加 Tag，逗号分隔"
                                )
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
        record_outputs = [
            records_state,
            preview_image,
            tags_text,
            nl_text,
            final_caption,
            current_index,
            status,
        ]
        edit_outputs = [tags_text, final_caption]
        autosave_outputs = [records_state, image_table, final_caption, current_index]

        open_button.click(
            events.open_folder,
            inputs=[dataset_path, metadata_location],
            outputs=open_outputs,
        )
        image_table.select(
            events.select_record,
            inputs=[records_state, current_index, tags_text, nl_text],
            outputs=record_outputs,
        )
        prev_button.click(
            events.previous_record,
            inputs=[records_state, current_index, tags_text, nl_text],
            outputs=record_outputs,
        )
        next_button.click(
            events.next_record,
            inputs=[records_state, current_index, tags_text, nl_text],
            outputs=record_outputs,
        )
        tags_text.input(
            events.autosave_current,
            inputs=[records_state, current_index, tags_text, nl_text],
            outputs=autosave_outputs,
        )
        nl_text.input(
            events.autosave_current,
            inputs=[records_state, current_index, tags_text, nl_text],
            outputs=autosave_outputs,
        )

        def show_toast(msg: str):
            if not msg:
                return
            if "失败" in msg or "错误" in msg or "异常" in msg:
                gr.Warning(msg)
            else:
                gr.Info(msg)

        status.change(show_toast, inputs=[status], outputs=None)

        apply_rules_button.click(
            events.apply_rules_to_current,
            inputs=[tags_text, nl_text],
            outputs=edit_outputs,
        )
        clear_tags_button.click(
            events.clear_tags, inputs=[tags_text, nl_text], outputs=edit_outputs
        )
        clear_nl_button.click(
            events.clear_nl,
            inputs=[tags_text, nl_text],
            outputs=[nl_text, final_caption],
        )

        generate_tag_button.click(
            events.generate_tag,
            inputs=[
                records_state,
                current_index,
                tag_model,
                workbench_tag_categories,
                tags_text,
                nl_text,
            ],
            outputs=open_outputs,
        )
        generate_nl_button.click(
            events.generate_nl,
            inputs=[
                records_state,
                current_index,
                nl_model,
                nl_endpoint,
                nl_model_name,
                nl_api_key,
                tags_text,
                nl_text,
                shuffle_tags,
                image_resize_mode,
            ],
            outputs=open_outputs,
        )
        generate_both_button.click(
            events.generate_tag_and_nl,
            inputs=[
                records_state,
                current_index,
                tag_model,
                workbench_tag_categories,
                nl_model,
                nl_endpoint,
                nl_model_name,
                nl_api_key,
                tags_text,
                nl_text,
                shuffle_tags,
                image_resize_mode,
            ],
            outputs=open_outputs,
        )
        save_current_button.click(
            events.save_current,
            inputs=[
                records_state,
                current_index,
                tags_text,
                nl_text,
                metadata_location,
            ],
            outputs=open_outputs,
        )
        save_all_button.click(
            events.save_all,
            inputs=[
                records_state,
                current_index,
                tags_text,
                nl_text,
                metadata_location,
            ],
            outputs=open_outputs,
        )

        batch_tag_button.click(
            events.batch_generate_tags,
            inputs=[
                records_state,
                current_index,
                tags_text,
                nl_text,
                tag_model,
                batch_tag_categories,
                skip_edited,
            ],
            outputs=open_outputs,
            concurrency_id="batch_generation",
            concurrency_limit=1,
        )
        batch_nl_button.click(
            events.batch_generate_nl,
            inputs=[
                records_state,
                current_index,
                tags_text,
                nl_text,
                nl_model,
                nl_endpoint,
                nl_model_name,
                nl_api_key,
                skip_edited,
                shuffle_tags,
                image_resize_mode,
            ],
            outputs=open_outputs,
            concurrency_id="batch_generation",
            concurrency_limit=1,
        )
        batch_both_button.click(
            events.batch_generate_both,
            inputs=[
                records_state,
                current_index,
                tags_text,
                nl_text,
                tag_model,
                batch_tag_categories,
                nl_model,
                nl_endpoint,
                nl_model_name,
                nl_api_key,
                skip_edited,
                shuffle_tags,
                image_resize_mode,
            ],
            outputs=open_outputs,
            concurrency_id="batch_generation",
            concurrency_limit=1,
        )
        stop_batch_button.click(
            events.stop_batch_generation,
            inputs=None,
            outputs=[status],
            queue=False,
        )
        delete_tag_button.click(
            events.batch_delete_tag,
            inputs=[records_state, current_index, tags_text, nl_text, delete_tag_text],
            outputs=open_outputs,
        )
        replace_tag_button.click(
            events.batch_replace_tag,
            inputs=[
                records_state,
                current_index,
                tags_text,
                nl_text,
                replace_old,
                replace_new,
            ],
            outputs=open_outputs,
        )
        add_tag_button.click(
            events.batch_add_tag,
            inputs=[
                records_state,
                current_index,
                tags_text,
                nl_text,
                add_tag_text,
                prepend_tag,
            ],
            outputs=open_outputs,
        )
        app.unload(events.cleanup_session_state)

    return app
